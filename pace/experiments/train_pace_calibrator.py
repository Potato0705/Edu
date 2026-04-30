"""Train the first runnable PACE-AES ordinal calibrator.

This script implements the smallest end-to-end Layer-2/3 loop:

1. Load the Layer-1 champion protocol (I*, static exemplars) for one prompt/fold.
2. Re-score train/val/test essays with the local Llama backend to obtain:
   - y_raw
   - hidden state h(x)
   - raw structured reasoning text
3. Merge h(x) with cached h(e_low/mid/high) to form anchor-relative evidence.
4. Train a CORAL-style ordinal calibrator on the train split.
5. Select by validation QWK and report train/val/test metrics.

This is intentionally a prompt/fold-local script. Once the method is stable we
can add a batch launcher across prompts and folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pace.calibration import (  # noqa: E402
    CalibratorConfig,
    CoralOrdinalCalibrator,
    DecodeSpec,
    MMDConfig,
    MMDProjectionHead,
    SegmentedOrdinalCalibrator,
    boundary_aware_mmd_separation_loss,
    calibrator_loss,
    decode_scores,
    fit_segment_edges,
    raw_scores_from_feature,
)
from pace.datasets import asap as asap_loader  # noqa: E402
from pace.evidence import build_evidence_vector  # noqa: E402
from pace.llm_backend import (  # noqa: E402
    LocalLlamaBackend,
    ScoringRequest,
    load_layer1_champion,
)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--logs-root", type=str, default="logs")
    p.add_argument("--prompt", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--cache-root", type=str, default="cache/pace_anchor_cache")
    p.add_argument("--out-dir", type=str, default="results/pace_calibration")
    p.add_argument("--gen", type=int, default=25)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-qwk", type=float, default=0.25)
    p.add_argument("--mmd-enable", action="store_true")
    p.add_argument("--mmd-feature-space", type=str, default="remb", choices=["remb"])
    p.add_argument("--mmd-scope", type=str, default="prompt_wise", choices=["prompt_wise"])
    p.add_argument("--mmd-band-mode", type=str, default="adjacent_only", choices=["adjacent_only"])
    p.add_argument(
        "--mmd-sample-mode",
        type=str,
        default="boundary_only",
        choices=["boundary_only", "all"],
    )
    p.add_argument("--mmd-kernel", type=str, default="rbf", choices=["rbf"])
    p.add_argument(
        "--mmd-sigma-mode",
        type=str,
        default="median_heuristic",
        choices=["median_heuristic", "fixed"],
    )
    p.add_argument("--mmd-sigma", type=float, default=1.0)
    p.add_argument("--mmd-margin", type=float, default=1.0)
    p.add_argument("--lambda-sep", type=float, default=0.0)
    p.add_argument(
        "--mmd-warmup-epochs",
        type=int,
        default=5,
        help="Delay MMD regularization for the first N epochs.",
    )
    p.add_argument("--mmd-min-samples-per-band", type=int, default=4)
    p.add_argument(
        "--mmd-boundary-mode",
        type=str,
        default="raw_boundary",
        choices=["uncertainty", "raw_boundary"],
    )
    p.add_argument("--mmd-uncertainty-threshold", type=float, default=0.0)
    p.add_argument("--mmd-raw-boundary-epsilon", type=float, default=1.0)
    p.add_argument(
        "--mmd-project-dim",
        type=int,
        default=0,
        help="Project r_emb to a smaller space before MMD. 0 disables projection.",
    )
    p.add_argument(
        "--mmd-project-dropout",
        type=float,
        default=0.0,
        help="Dropout inside the MMD projector.",
    )
    p.add_argument(
        "--mmd-num-bands",
        type=int,
        default=0,
        help="Band count for MMD grouping. <=0 means infer from anchor count.",
    )
    p.add_argument(
        "--num-segments",
        type=int,
        default=1,
        help=(
            "Number of raw-score segments for the piecewise calibrator. "
            "Use 1 for the original global calibrator."
        ),
    )
    p.add_argument(
        "--decode-mode",
        type=str,
        default="threshold",
        choices=["threshold", "expected_round", "blend_round"],
        help=(
            "How to decode final integer scores from the ordinal calibrator. "
            "'threshold' reproduces the original CORAL hard decode; "
            "'expected_round' rounds the expected score; "
            "'blend_round' conservatively blends expected score toward raw score."
        ),
    )
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=1.0,
        help="Used only when --decode-mode=blend_round. 0 keeps raw score, 1 uses full expected-score correction.",
    )
    p.add_argument(
        "--max-delta-frac",
        type=float,
        default=1.0,
        help=(
            "Used only when --decode-mode=blend_round. Caps |y_pred - y_raw| to "
            "this fraction of the prompt score span. 1.0 disables clipping."
        ),
    )
    p.add_argument(
        "--auto-select-decode",
        action="store_true",
        help=(
            "Select decode mode / blend parameters on the validation split "
            "instead of trusting a single fixed decode setting."
        ),
    )
    p.add_argument(
        "--decode-search-alphas",
        type=float,
        nargs="+",
        default=[0.35, 0.50, 0.65],
        help="Candidate blend_alpha values when --auto-select-decode is enabled.",
    )
    p.add_argument(
        "--decode-search-fracs",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
        help="Candidate max_delta_frac values when --auto-select-decode is enabled.",
    )
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help=(
            "Save evidence extraction progress every N essays per split. "
            "Set to 0 to disable split-level checkpoints."
        ),
    )
    p.add_argument(
        "--no-resume-evidence",
        action="store_true",
        help="Do not resume from split-level evidence checkpoints.",
    )
    p.add_argument(
        "--rebuild-evidence",
        action="store_true",
        help="Ignore saved evidence_cache.pt and rebuild by running the local Llama scorer.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve protocol/splits/cache paths only; do not load the model.",
    )
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_exp_dir(logs_root: Path, prompt: int, fold: int) -> Path:
    candidates: List[Path] = []
    for d in sorted(logs_root.glob(f"exp_*_fold{fold}")):
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        if int(cfg.get("data", {}).get("essay_set", -1)) == prompt:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No exp dir under {logs_root} for prompt={prompt} fold={fold}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _resolve_static_exemplars(train_set: List[Dict], exemplar_ids: List[int]) -> List[Dict]:
    id_map = {int(e["essay_id"]): e for e in train_set}
    resolved = [id_map[int(i)] for i in exemplar_ids if int(i) in id_map]
    if not resolved:
        raise RuntimeError(
            "Champion static exemplars could not be resolved against the train split."
        )
    return resolved


def _maybe_limit(items: List[Dict], limit: int) -> List[Dict]:
    if limit and limit > 0:
        return items[:limit]
    return items


def _load_anchor_entry(
    cache_root: Path,
    prompt: int,
    fold: int,
    backend_signature: str,
) -> Dict:
    path = cache_root / f"asap_p{prompt}_fold{fold}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Anchor cache missing: {path}. Run pace.experiments.build_anchor_cache first."
        )
    entry = torch.load(path, map_location="cpu")
    if entry.get("backend_signature") != backend_signature:
        raise RuntimeError(
            "Anchor cache signature mismatch. Rebuild the anchor cache with the same backend "
            "settings as the calibrator run."
        )
    return entry


def _split_payload_from_parts(
    rows: List[Dict],
    zs: List[torch.Tensor],
    ys: List[int],
) -> Dict:
    if zs:
        z = torch.stack(zs, dim=0)
    else:
        z = torch.empty((0, 0), dtype=torch.float32)
    return {
        "rows": rows,
        "essay_ids": torch.tensor([r["essay_id"] for r in rows], dtype=torch.long),
        "y_true": torch.tensor(ys, dtype=torch.long),
        "y_raw": torch.tensor([r["y_raw"] for r in rows], dtype=torch.long),
        "z": z,
    }


def _infer_num_anchors_from_zdim(z_dim: int) -> int:
    # z = [raw(1); r_emb; r_s(11); f_o(11); u(6)]
    structured_dim = 1 + 11 + 11 + 6
    tail_free = z_dim - structured_dim
    for n_anchors in range(3, 9):
        numer = tail_free - (2 * n_anchors)
        denom = n_anchors + 1
        if numer > 0 and numer % denom == 0:
            hidden_dim = numer // denom
            if 256 <= hidden_dim <= 16384:
                return n_anchors
    return 3


def _hydrate_split_aux(
    split_payload: Dict,
    *,
    prompt_id: int,
    num_anchors: int,
) -> None:
    z = split_payload["z"].float()
    if z.numel() == 0:
        split_payload["r_emb"] = torch.empty((0, 0), dtype=torch.float32)
        split_payload["u"] = torch.empty((0, 0), dtype=torch.float32)
        split_payload["uncertainty_scalar"] = torch.empty((0,), dtype=torch.float32)
        split_payload["prompt_id"] = torch.empty((0,), dtype=torch.long)
        return
    u_dim = 6
    r_s_dim = 11
    f_o_dim = 11
    fixed_tail = u_dim + r_s_dim + f_o_dim
    r_emb_dim = int(z.shape[1] - 1 - fixed_tail)
    if r_emb_dim <= 0:
        raise RuntimeError(f"Cannot infer r_emb_dim from z shape {tuple(z.shape)}")
    split_payload["r_emb"] = z[:, 1 : 1 + r_emb_dim]
    split_payload["u"] = z[:, -u_dim:]
    u_no_raw = split_payload["u"][:, :-1] if u_dim > 1 else split_payload["u"]
    split_payload["uncertainty_scalar"] = u_no_raw.norm(p=2, dim=1)
    split_payload["prompt_id"] = torch.full(
        (z.shape[0],),
        int(prompt_id),
        dtype=torch.long,
    )


def _save_split_checkpoint(path: Path, meta: Dict, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"meta": meta, "payload": payload}, tmp_path)
    tmp_path.replace(path)


def _load_split_checkpoint(path: Path, expected_meta: Dict) -> Dict | None:
    if not path.exists():
        return None
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[pace] WARN could not load split checkpoint {path}: {exc}")
        return None
    meta = obj.get("meta", {})
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            print(
                f"[pace] Ignore split checkpoint {path}: "
                f"meta[{key!r}]={meta.get(key)!r} expected {expected!r}"
            )
            return None
    payload = obj.get("payload")
    if not isinstance(payload, dict) or "rows" not in payload or "z" not in payload:
        print(f"[pace] Ignore split checkpoint {path}: invalid payload")
        return None
    return payload


def _build_split_payload(
    *,
    split_name: str,
    items: List[Dict],
    instruction: str,
    static_exemplars: List[Dict],
    score_min: int,
    score_max: int,
    backend: LocalLlamaBackend,
    anchor_entry: Dict,
    checkpoint_path: Path | None = None,
    checkpoint_meta: Dict | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> Dict:
    rows: List[Dict] = []
    zs: List[torch.Tensor] = []
    ys: List[int] = []
    processed_ids = set()
    print(f"[pace] Building evidence for {split_name}: n={len(items)}")

    if resume and checkpoint_path is not None and checkpoint_meta is not None:
        partial = _load_split_checkpoint(checkpoint_path, checkpoint_meta)
        if partial is not None:
            rows = list(partial["rows"])
            z_tensor = partial["z"]
            zs = [t.detach().clone() for t in z_tensor.unbind(0)]
            ys = [int(v) for v in partial["y_true"].cpu().tolist()]
            processed_ids = {int(r["essay_id"]) for r in rows}
            print(
                f"[pace] Resumed {split_name} evidence checkpoint: "
                f"{len(rows)}/{len(items)} from {checkpoint_path}"
            )
            if len(rows) >= len(items):
                return _split_payload_from_parts(rows, zs, ys)

    for idx, item in enumerate(items, start=1):
        essay_id = int(item["essay_id"])
        if essay_id in processed_ids:
            if idx == 1 or idx % 25 == 0 or idx == len(items):
                print(
                    f"[pace] {split_name} {idx}/{len(items)} "
                    f"essay={essay_id} already cached"
                )
            continue
        req = ScoringRequest(
            essay_id=essay_id,
            essay_text=item["essay_text"],
            instruction=instruction,
            static_exemplars=static_exemplars,
            score_min=score_min,
            score_max=score_max,
            dynamic_ex="(None)",
        )
        res = backend.score(req)
        bundle = build_evidence_vector(
            essay_text=item["essay_text"],
            result=res,
            anchor_entry=type("AnchorEntryView", (), anchor_entry),
            score_min=score_min,
            score_max=score_max,
        )
        zs.append(bundle.z)
        ys.append(int(item["domain1_score"]))
        rows.append(
            {
                "split": split_name,
                "essay_id": essay_id,
                "y_true": int(item["domain1_score"]),
                "y_raw": int(res.y_raw),
                "wallclock_sec": float(res.wallclock_sec),
            }
        )
        if idx == 1 or idx % 25 == 0 or idx == len(items):
            print(
                f"[pace] {split_name} {idx}/{len(items)} "
                f"essay={item['essay_id']} y_true={item['domain1_score']} y_raw={res.y_raw}"
            )
        if (
            checkpoint_path is not None
            and checkpoint_meta is not None
            and checkpoint_every > 0
            and (len(rows) % checkpoint_every == 0 or idx == len(items))
        ):
            payload = _split_payload_from_parts(rows, zs, ys)
            _save_split_checkpoint(checkpoint_path, checkpoint_meta, payload)
            print(
                f"[pace] Saved {split_name} evidence checkpoint: "
                f"{len(rows)}/{len(items)} -> {checkpoint_path}"
            )

    payload = _split_payload_from_parts(rows, zs, ys)
    if checkpoint_path is not None and checkpoint_meta is not None and checkpoint_every > 0:
        _save_split_checkpoint(checkpoint_path, checkpoint_meta, payload)
        print(
            f"[pace] Saved {split_name} evidence checkpoint: "
            f"{len(rows)}/{len(items)} -> {checkpoint_path}"
        )
    return payload


def _assert_complete_split(split_name: str, payload: Dict, expected_count: int) -> None:
    actual = int(payload["z"].shape[0])
    if actual != expected_count:
        raise RuntimeError(
            f"Incomplete evidence for split={split_name}: "
            f"expected {expected_count}, got {actual}."
        )


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def _resolve_decode_spec(
    *,
    decode_mode: str,
    blend_alpha: float,
    max_delta_frac: float,
    max_delta: float | None,
) -> DecodeSpec:
    return DecodeSpec(
        mode=decode_mode,
        blend_alpha=float(blend_alpha),
        max_delta=max_delta,
        max_delta_frac=float(max_delta_frac),
    )


def _build_decode_candidates(
    args: argparse.Namespace,
    *,
    score_span: int,
    max_delta: float | None,
) -> List[DecodeSpec]:
    fixed = _resolve_decode_spec(
        decode_mode=args.decode_mode,
        blend_alpha=args.blend_alpha,
        max_delta_frac=args.max_delta_frac,
        max_delta=max_delta,
    )
    if not args.auto_select_decode:
        return [fixed]

    specs: List[DecodeSpec] = [
        DecodeSpec("threshold", 1.0, None, 1.0),
        DecodeSpec("expected_round", 1.0, None, 1.0),
        fixed,
    ]
    for alpha in args.decode_search_alphas:
        for frac in args.decode_search_fracs:
            capped = None if frac >= 1.0 else float(frac) * float(max(1, score_span))
            specs.append(
                DecodeSpec(
                    mode="blend_round",
                    blend_alpha=float(alpha),
                    max_delta=capped,
                    max_delta_frac=float(frac),
                )
            )

    dedup: List[DecodeSpec] = []
    seen = set()
    for spec in specs:
        key = (
            spec.mode,
            round(float(spec.blend_alpha), 6),
            None if spec.max_delta is None else round(float(spec.max_delta), 6),
            round(float(spec.max_delta_frac), 6),
        )
        if key not in seen:
            seen.add(key)
            dedup.append(spec)
    return dedup


def _decode_with_spec(
    *,
    z: torch.Tensor,
    out: Dict[str, torch.Tensor],
    score_min: int,
    score_max: int,
    spec: DecodeSpec,
) -> torch.Tensor:
    return decode_scores(
        raw_score=raw_scores_from_feature(z, score_min, score_max),
        expected_score=out["expected_score"],
        cum_probs=out["cum_probs"],
        score_min=score_min,
        score_max=score_max,
        decode_mode=spec.mode,
        blend_alpha=spec.blend_alpha,
        max_delta=spec.max_delta,
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    z: torch.Tensor,
    y_true: torch.Tensor,
    *,
    decode_candidates: List[DecodeSpec],
    mmd_cfg: MMDConfig | None = None,
    mmd_projector: torch.nn.Module | None = None,
    r_emb: torch.Tensor | None = None,
    y_raw: torch.Tensor | None = None,
    prompt_ids: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.eval()
    if mmd_projector is not None:
        mmd_projector.eval()
    out = model(z)
    total_loss, loss_parts = calibrator_loss(model, z, y_true)
    sep_loss = torch.tensor(0.0, device=z.device)
    sep_stats: Dict[str, object] = {
        "loss_sep": 0.0,
        "num_prompts_used_for_mmd": 0,
        "num_adjacent_pairs_used": 0,
        "num_samples_used_for_mmd": 0,
        "avg_mmd_value": 0.0,
        "avg_margin_gap": 0.0,
        "per_prompt_mmd": {},
    }
    if (
        mmd_cfg is not None
        and mmd_cfg.enable
        and mmd_cfg.lambda_sep > 0.0
        and r_emb is not None
        and y_raw is not None
        and prompt_ids is not None
        and uncertainty is not None
    ):
        sep_features = mmd_projector(r_emb) if mmd_projector is not None else r_emb
        sep_loss, sep_stats = boundary_aware_mmd_separation_loss(
            features=sep_features,
            prompt_ids=prompt_ids,
            y_true=y_true,
            y_raw=y_raw,
            uncertainty=uncertainty,
            score_min=model.config.score_min,
            score_max=model.config.score_max,
            config=mmd_cfg,
        )
        total_loss = total_loss + float(mmd_cfg.lambda_sep) * sep_loss
    y_true_np = y_true.cpu().numpy()
    best = None
    for spec in decode_candidates:
        y_pred = _decode_with_spec(
            z=z,
            out=out,
            score_min=model.config.score_min,
            score_max=model.config.score_max,
            spec=spec,
        )
        y_pred_np = y_pred.cpu().numpy()
        qwk = _qwk(y_true_np, y_pred_np)
        mae = float(mean_absolute_error(y_true_np, y_pred_np))
        acc = float((y_true == y_pred).float().mean().item())
        candidate = {
            "spec": spec,
            "qwk": qwk,
            "mae": mae,
            "acc": acc,
            "y_pred": y_pred_np.tolist(),
        }
        key = (qwk, -mae, acc)
        if best is None or key > best["key"]:
            best = {"key": key, "payload": candidate}
    assert best is not None
    chosen = best["payload"]
    return {
        "loss_total": float(total_loss.cpu()),
        "loss_ord": loss_parts["loss_ord"],
        "loss_qwk": loss_parts["loss_qwk"],
        "soft_qwk_score": loss_parts["soft_qwk_score"],
        "loss_reg": loss_parts["loss_reg"],
        "loss_sep": float(sep_stats["loss_sep"]),
        "num_prompts_used_for_mmd": int(sep_stats["num_prompts_used_for_mmd"]),
        "num_adjacent_pairs_used": int(sep_stats["num_adjacent_pairs_used"]),
        "num_samples_used_for_mmd": int(sep_stats["num_samples_used_for_mmd"]),
        "avg_mmd_value": float(sep_stats["avg_mmd_value"]),
        "avg_margin_gap": float(sep_stats["avg_margin_gap"]),
        "qwk": chosen["qwk"],
        "mae": chosen["mae"],
        "acc": chosen["acc"],
        "y_pred": chosen["y_pred"],
        "expected_score": out["expected_score"].cpu().tolist(),
        "selected_decode": {
            "mode": chosen["spec"].mode,
            "blend_alpha": float(chosen["spec"].blend_alpha),
            "max_delta": (
                None if chosen["spec"].max_delta is None
                else float(chosen["spec"].max_delta)
            ),
            "max_delta_frac": float(chosen["spec"].max_delta_frac),
        },
        "per_prompt_mmd": sep_stats["per_prompt_mmd"],
        "segment_ids": out.get("segment_ids", torch.zeros(z.shape[0], dtype=torch.long)).cpu().tolist(),
    }


def _standardize_splits(splits: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
    train_tail = splits["train"]["z"][:, 1:]
    mean = train_tail.mean(dim=0)
    std = train_tail.std(dim=0, unbiased=False)
    std = torch.where(std < 1.0e-6, torch.ones_like(std), std)
    return {"mean": mean, "std": std}


def _apply_standardize(z: torch.Tensor, scaler: Dict[str, torch.Tensor]) -> torch.Tensor:
    if z.shape[1] <= 1:
        return z
    head = z[:, :1]
    tail = (z[:, 1:] - scaler["mean"]) / scaler["std"]
    return torch.cat([head, tail], dim=1)


def main() -> int:
    args = _parse_cli()
    _set_seed(args.seed)

    logs_root = Path(args.logs_root)
    out_dir = Path(args.out_dir) / f"p{args.prompt}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_cache_path = out_dir / "evidence_cache.pt"

    exp_dir = _find_exp_dir(logs_root, args.prompt, args.fold)
    champion = load_layer1_champion(exp_dir, gen=args.gen)
    cfg = champion["config"]
    score_min = int(cfg["data"]["score_min"])
    score_max = int(cfg["data"]["score_max"])

    all_data = asap_loader.load_asap(args.prompt, repo_root=_REPO_ROOT)
    train_set, val_set, test_set = asap_loader.split_train_val_test_for_fold(
        all_data, args.fold
    )
    train_set = _maybe_limit(train_set, args.limit_train)
    val_set = _maybe_limit(val_set, args.limit_val)
    test_set = _maybe_limit(test_set, args.limit_test)

    static_exemplars = _resolve_static_exemplars(train_set, champion["static_exemplar_ids"])

    print(f"[pace] exp_dir={exp_dir}")
    print(
        f"[pace] splits train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"score=[{score_min},{score_max}]"
    )

    if args.dry_run:
        print("[pace] --dry-run complete. Protocol and split resolution succeeded.")
        return 0

    if evidence_cache_path.exists() and not args.rebuild_evidence:
        payload = torch.load(evidence_cache_path, map_location="cpu")
        print(f"[pace] Loaded evidence cache: {evidence_cache_path}")
        anchor_cache_path = Path(args.cache_root) / f"asap_p{args.prompt}_fold{args.fold}.pt"
        if anchor_cache_path.exists():
            try:
                anchor_obj = torch.load(anchor_cache_path, map_location="cpu")
                current_num_anchors = int(anchor_obj["hidden"].shape[0])
                cached_num_anchors = int(payload.get("meta", {}).get("num_anchors", -1))
                if cached_num_anchors > 0 and cached_num_anchors != current_num_anchors:
                    raise RuntimeError(
                        "Evidence cache anchor count does not match the current anchor cache. "
                        "Re-run with --rebuild-evidence."
                    )
                if cached_num_anchors <= 0:
                    print(
                        "[pace] WARN evidence cache does not record num_anchors; "
                        "if you recently changed the anchor cache, rebuild evidence."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[pace] WARN could not verify anchor count against cache: {exc}")
    else:
        backend = LocalLlamaBackend(
            cfg,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
        )
        anchor_entry = _load_anchor_entry(
            Path(args.cache_root),
            args.prompt,
            args.fold,
            backend.signature(),
        )
        base_meta = {
            "prompt": args.prompt,
            "fold": args.fold,
            "score_min": score_min,
            "score_max": score_max,
            "num_anchors": int(anchor_entry["hidden"].shape[0]),
            "model_path": args.model_path,
            "backend_signature": backend.signature(),
            "exp_dir": str(exp_dir),
            "input_instruction_len": len(champion["instruction"]),
        }
        resume_split_checkpoints = not args.no_resume_evidence and not args.rebuild_evidence
        payload = {
            "meta": base_meta,
            "splits": {},
        }
        split_items = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }
        for split_name, items in split_items.items():
            checkpoint_meta = {
                **base_meta,
                "checkpoint_version": 1,
                "split": split_name,
                "item_count": len(items),
            }
            split_payload = _build_split_payload(
                split_name=split_name,
                items=items,
                instruction=champion["instruction"],
                static_exemplars=static_exemplars,
                score_min=score_min,
                score_max=score_max,
                backend=backend,
                anchor_entry=anchor_entry,
                checkpoint_path=out_dir / f"evidence_{split_name}.partial.pt",
                checkpoint_meta=checkpoint_meta,
                checkpoint_every=args.checkpoint_every,
                resume=resume_split_checkpoints,
            )
            _assert_complete_split(split_name, split_payload, len(items))
            payload["splits"][split_name] = split_payload
        torch.save(payload, evidence_cache_path)
        print(f"[pace] Saved evidence cache: {evidence_cache_path}")

    splits = payload["splits"]
    payload_num_anchors = int(payload.get("meta", {}).get("num_anchors", 0))
    if payload_num_anchors <= 0:
        payload_num_anchors = _infer_num_anchors_from_zdim(int(splits["train"]["z"].shape[1]))
    for split in ("train", "val", "test"):
        _hydrate_split_aux(
            splits[split],
            prompt_id=args.prompt,
            num_anchors=payload_num_anchors,
        )
    scaler = _standardize_splits(splits)
    for split in ("train", "val", "test"):
        splits[split]["z_scaled"] = _apply_standardize(splits[split]["z"].float(), scaler)

    input_dim = int(splits["train"]["z_scaled"].shape[1])
    score_span = max(1, score_max - score_min)
    max_delta = (
        None
        if args.max_delta_frac >= 1.0
        else float(args.max_delta_frac) * float(score_span)
    )
    mmd_num_bands = int(args.mmd_num_bands) if int(args.mmd_num_bands) > 0 else int(payload_num_anchors)
    mmd_cfg = MMDConfig(
        enable=bool(args.mmd_enable),
        feature_space=str(args.mmd_feature_space),
        scope=str(args.mmd_scope),
        band_mode=str(args.mmd_band_mode),
        sample_mode=str(args.mmd_sample_mode),
        warmup_epochs=int(args.mmd_warmup_epochs),
        kernel=str(args.mmd_kernel),
        sigma_mode=str(args.mmd_sigma_mode),
        sigma=float(args.mmd_sigma),
        margin=float(args.mmd_margin),
        lambda_sep=float(args.lambda_sep),
        min_samples_per_band=int(args.mmd_min_samples_per_band),
        boundary_mode=str(args.mmd_boundary_mode),
        uncertainty_threshold=float(args.mmd_uncertainty_threshold),
        raw_boundary_epsilon=float(args.mmd_raw_boundary_epsilon),
        num_bands=int(mmd_num_bands),
        project_dim=int(args.mmd_project_dim),
        project_dropout=float(args.mmd_project_dropout),
    )
    decode_candidates = _build_decode_candidates(
        args,
        score_span=score_span,
        max_delta=max_delta,
    )
    cal_cfg = CalibratorConfig(
        input_dim=input_dim,
        score_min=score_min,
        score_max=score_max,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_qwk=args.lambda_qwk,
        lambda_reg=args.weight_decay,
    )
    segment_edges = (
        fit_segment_edges(splits["train"]["z_scaled"][:, 0], args.num_segments)
        if args.num_segments > 1
        else []
    )
    if args.num_segments > 1:
        model = SegmentedOrdinalCalibrator(cal_cfg, segment_edges)
    else:
        model = CoralOrdinalCalibrator(cal_cfg)
    mmd_projector: torch.nn.Module | None = None
    if mmd_cfg.enable and mmd_cfg.project_dim > 0:
        r_emb_dim = int(splits["train"]["r_emb"].shape[1])
        mmd_projector = MMDProjectionHead(
            input_dim=r_emb_dim,
            output_dim=int(mmd_cfg.project_dim),
            dropout=float(mmd_cfg.project_dropout),
        )
    optim_params = list(model.parameters())
    if mmd_projector is not None:
        optim_params.extend(list(mmd_projector.parameters()))
    opt = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=0.0)

    train_loader = DataLoader(
        TensorDataset(
            splits["train"]["z_scaled"],
            splits["train"]["y_true"],
            splits["train"]["r_emb"],
            splits["train"]["y_raw"].float(),
            splits["train"]["prompt_id"],
            splits["train"]["uncertainty_scalar"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_model_state = None
    best_projector_state = None
    best_epoch = -1
    best_val_qwk = -1.0e9
    best_decode_spec: DecodeSpec | None = None
    history: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if mmd_projector is not None:
            mmd_projector.train()
        batch_losses: List[float] = []
        batch_sep_losses: List[float] = []
        batch_mmd_values: List[float] = []
        batch_margin_gaps: List[float] = []
        batch_pairs_used: List[int] = []
        batch_prompts_used: List[int] = []
        batch_samples_used: List[int] = []
        mmd_active = (
            mmd_cfg.enable
            and mmd_cfg.lambda_sep > 0.0
            and epoch > int(mmd_cfg.warmup_epochs)
        )
        for z_batch, y_batch, r_emb_batch, y_raw_batch, prompt_batch, unc_batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, loss_parts = calibrator_loss(model, z_batch, y_batch)
            if mmd_active:
                sep_features = (
                    mmd_projector(r_emb_batch)
                    if mmd_projector is not None
                    else r_emb_batch
                )
                sep_loss, sep_stats = boundary_aware_mmd_separation_loss(
                    features=sep_features,
                    prompt_ids=prompt_batch,
                    y_true=y_batch,
                    y_raw=y_raw_batch,
                    uncertainty=unc_batch,
                    score_min=score_min,
                    score_max=score_max,
                    config=mmd_cfg,
                )
                loss = loss + float(mmd_cfg.lambda_sep) * sep_loss
                batch_sep_losses.append(float(sep_stats["loss_sep"]))
                batch_mmd_values.append(float(sep_stats["avg_mmd_value"]))
                batch_margin_gaps.append(float(sep_stats["avg_margin_gap"]))
                batch_pairs_used.append(int(sep_stats["num_adjacent_pairs_used"]))
                batch_prompts_used.append(int(sep_stats["num_prompts_used_for_mmd"]))
                batch_samples_used.append(int(sep_stats["num_samples_used_for_mmd"]))
            else:
                batch_sep_losses.append(0.0)
                batch_mmd_values.append(0.0)
                batch_margin_gaps.append(0.0)
                batch_pairs_used.append(0)
                batch_prompts_used.append(0)
                batch_samples_used.append(0)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))

        val_metrics = _evaluate(
            model,
            splits["val"]["z_scaled"],
            splits["val"]["y_true"],
            decode_candidates=decode_candidates,
            mmd_cfg=mmd_cfg,
            mmd_projector=mmd_projector,
            r_emb=splits["val"]["r_emb"],
            y_raw=splits["val"]["y_raw"].float(),
            prompt_ids=splits["val"]["prompt_id"],
            uncertainty=splits["val"]["uncertainty_scalar"],
        )
        chosen = val_metrics["selected_decode"]
        chosen_spec = DecodeSpec(
            mode=chosen["mode"],
            blend_alpha=float(chosen["blend_alpha"]),
            max_delta=chosen["max_delta"],
            max_delta_frac=float(chosen["max_delta_frac"]),
        )
        train_metrics = _evaluate(
            model,
            splits["train"]["z_scaled"],
            splits["train"]["y_true"],
            decode_candidates=[chosen_spec],
            mmd_cfg=mmd_cfg,
            mmd_projector=mmd_projector,
            r_emb=splits["train"]["r_emb"],
            y_raw=splits["train"]["y_raw"].float(),
            prompt_ids=splits["train"]["prompt_id"],
            uncertainty=splits["train"]["uncertainty_scalar"],
        )
        row = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "train_loss_sep": float(np.mean(batch_sep_losses)) if batch_sep_losses else 0.0,
            "train_num_prompts_used_for_mmd": float(np.mean(batch_prompts_used)) if batch_prompts_used else 0.0,
            "train_num_adjacent_pairs_used": float(np.mean(batch_pairs_used)) if batch_pairs_used else 0.0,
            "train_num_samples_used_for_mmd": float(np.mean(batch_samples_used)) if batch_samples_used else 0.0,
            "train_avg_mmd_value": float(np.mean(batch_mmd_values)) if batch_mmd_values else 0.0,
            "train_avg_margin_gap": float(np.mean(batch_margin_gaps)) if batch_margin_gaps else 0.0,
            "train_qwk": train_metrics["qwk"],
            "train_mae": train_metrics["mae"],
            "val_qwk": val_metrics["qwk"],
            "val_mae": val_metrics["mae"],
            "val_loss_total": val_metrics["loss_total"],
            "val_loss_sep": val_metrics["loss_sep"],
            "val_num_prompts_used_for_mmd": val_metrics["num_prompts_used_for_mmd"],
            "val_num_adjacent_pairs_used": val_metrics["num_adjacent_pairs_used"],
            "val_num_samples_used_for_mmd": val_metrics["num_samples_used_for_mmd"],
            "val_avg_mmd_value": val_metrics["avg_mmd_value"],
            "val_avg_margin_gap": val_metrics["avg_margin_gap"],
            "decode_mode": chosen_spec.mode,
            "blend_alpha": chosen_spec.blend_alpha,
            "max_delta_frac": chosen_spec.max_delta_frac,
            "mmd_active": int(mmd_active),
        }
        history.append(row)
        print(
            f"[pace] epoch={epoch:02d} "
            f"train_qwk={train_metrics['qwk']:.4f} val_qwk={val_metrics['qwk']:.4f} "
            f"train_mae={train_metrics['mae']:.3f} val_mae={val_metrics['mae']:.3f} "
            f"train_sep={row['train_loss_sep']:.4f} val_sep={val_metrics['loss_sep']:.4f} "
            f"decode={chosen_spec.mode}"
        )
        if val_metrics["qwk"] > best_val_qwk:
            best_val_qwk = val_metrics["qwk"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_projector_state = (
                copy.deepcopy(mmd_projector.state_dict())
                if mmd_projector is not None
                else None
            )
            best_decode_spec = chosen_spec

    if best_model_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_model_state)
    if mmd_projector is not None and best_projector_state is not None:
        mmd_projector.load_state_dict(best_projector_state)
    if best_decode_spec is None:
        raise RuntimeError("Training did not retain a decode configuration.")

    val_metrics = _evaluate(
        model,
        splits["val"]["z_scaled"],
        splits["val"]["y_true"],
        decode_candidates=decode_candidates,
        mmd_cfg=mmd_cfg,
        mmd_projector=mmd_projector,
        r_emb=splits["val"]["r_emb"],
        y_raw=splits["val"]["y_raw"].float(),
        prompt_ids=splits["val"]["prompt_id"],
        uncertainty=splits["val"]["uncertainty_scalar"],
    )
    chosen = val_metrics["selected_decode"]
    selected_decode = DecodeSpec(
        mode=chosen["mode"],
        blend_alpha=float(chosen["blend_alpha"]),
        max_delta=chosen["max_delta"],
        max_delta_frac=float(chosen["max_delta_frac"]),
    )
    train_metrics = _evaluate(
        model,
        splits["train"]["z_scaled"],
        splits["train"]["y_true"],
        decode_candidates=[selected_decode],
        mmd_cfg=mmd_cfg,
        mmd_projector=mmd_projector,
        r_emb=splits["train"]["r_emb"],
        y_raw=splits["train"]["y_raw"].float(),
        prompt_ids=splits["train"]["prompt_id"],
        uncertainty=splits["train"]["uncertainty_scalar"],
    )
    test_metrics = _evaluate(
        model,
        splits["test"]["z_scaled"],
        splits["test"]["y_true"],
        decode_candidates=[selected_decode],
        mmd_cfg=mmd_cfg,
        mmd_projector=mmd_projector,
        r_emb=splits["test"]["r_emb"],
        y_raw=splits["test"]["y_raw"].float(),
        prompt_ids=splits["test"]["prompt_id"],
        uncertainty=splits["test"]["uncertainty_scalar"],
    )

    threshold_values = model.thresholds().detach().cpu().tolist()
    if hasattr(model, "raw_score_scales"):
        raw_score_scale = model.raw_score_scales().cpu().tolist()
    else:
        raw_score_scale = float(model.raw_score_scale.detach().cpu())
    summary = {
        "prompt": args.prompt,
        "fold": args.fold,
        "score_min": score_min,
        "score_max": score_max,
        "best_epoch": best_epoch,
        "input_dim": input_dim,
        "model_type": model.__class__.__name__,
        "num_anchors": int(payload.get("meta", {}).get("num_anchors", 0)),
        "num_segments": int(len(segment_edges) + 1),
        "segment_edges": [float(x) for x in segment_edges],
        "train_metrics": {k: v for k, v in train_metrics.items() if k not in {"y_pred", "expected_score", "segment_ids", "selected_decode"}},
        "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_pred", "expected_score", "segment_ids", "selected_decode"}},
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in {"y_pred", "expected_score", "segment_ids", "selected_decode"}},
        "thresholds": threshold_values,
        "raw_score_scale": raw_score_scale,
        "decode_mode": selected_decode.mode,
        "blend_alpha": float(selected_decode.blend_alpha),
        "max_delta_frac": float(selected_decode.max_delta_frac),
        "selected_decode": val_metrics["selected_decode"],
        "mmd": {
            "enable": mmd_cfg.enable,
            "feature_space": mmd_cfg.feature_space,
            "scope": mmd_cfg.scope,
            "band_mode": mmd_cfg.band_mode,
            "sample_mode": mmd_cfg.sample_mode,
            "kernel": mmd_cfg.kernel,
            "sigma_mode": mmd_cfg.sigma_mode,
            "sigma": mmd_cfg.sigma,
            "margin": mmd_cfg.margin,
            "lambda_sep": mmd_cfg.lambda_sep,
            "warmup_epochs": mmd_cfg.warmup_epochs,
            "min_samples_per_band": mmd_cfg.min_samples_per_band,
            "boundary_mode": mmd_cfg.boundary_mode,
            "uncertainty_threshold": mmd_cfg.uncertainty_threshold,
            "raw_boundary_epsilon": mmd_cfg.raw_boundary_epsilon,
            "num_bands": mmd_cfg.num_bands,
            "project_dim": mmd_cfg.project_dim,
            "project_dropout": mmd_cfg.project_dropout,
        },
        "evidence_cache": str(evidence_cache_path),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    per_essay_rows: List[Dict] = []
    for split_name, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ):
        y_true = splits[split_name]["y_true"].cpu().tolist()
        y_raw = splits[split_name]["y_raw"].cpu().tolist()
        essay_ids = splits[split_name]["essay_ids"].cpu().tolist()
        for essay_id, yt, yr, yp, ye, seg_id in zip(
            essay_ids,
            y_true,
            y_raw,
            metrics["y_pred"],
            metrics["expected_score"],
            metrics["segment_ids"],
        ):
            per_essay_rows.append(
                {
                    "split": split_name,
                    "essay_id": int(essay_id),
                    "y_true": int(yt),
                    "y_raw": int(yr),
                    "y_pred": int(yp),
                    "expected_score": float(ye),
                    "segment_id": int(seg_id),
                    "abs_err_raw": abs(int(yr) - int(yt)),
                    "abs_err_pace": abs(int(yp) - int(yt)),
                }
            )
    pd.DataFrame(per_essay_rows).to_csv(out_dir / "per_essay.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": model.__class__.__name__,
            "config": cal_cfg.__dict__,
            "scaler_mean": scaler["mean"],
            "scaler_std": scaler["std"],
            "segment_edges": [float(x) for x in segment_edges],
            "selected_decode": summary["selected_decode"],
            "mmd": summary["mmd"],
            "mmd_projector_state_dict": (
                mmd_projector.state_dict() if mmd_projector is not None else None
            ),
            "summary": summary,
        },
        out_dir / "calibrator.pt",
    )

    print(f"[pace] Wrote {out_dir / 'summary.json'}")
    print(f"[pace] Wrote {out_dir / 'history.csv'}")
    print(f"[pace] Wrote {out_dir / 'per_essay.csv'}")
    print(f"[pace] Wrote {out_dir / 'calibrator.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
