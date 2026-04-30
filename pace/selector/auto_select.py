"""Automatic prompt-aware recipe selection for PACE Layer-2."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from pace.calibration import (
    CalibratorConfig,
    CoralOrdinalCalibrator,
    MMDConfig,
    MMDProjectionHead,
    boundary_aware_mmd_separation_loss,
    calibrator_loss,
    decode_scores,
    raw_scores_from_feature,
)
from pace.selector.diagnostics import (
    LoadedEvidence,
    compute_prompt_diagnostics,
    concat_splits,
    load_evidence_cache,
)
from pace.selector.recipe_library import Recipe, get_recipe_library, manual_prompt_recipe_v1
from pace.selector.rule_gate import RuleGateConfig, apply_rule_gate


@dataclass
class SelectorTrainConfig:
    epochs: int = 25
    batch_size: int = 32
    hidden_dim: int = 512
    dropout: float = 0.1
    weight_decay: float = 1e-4
    seed: int = 13
    mmd_warmup_epochs: int = 3
    mmd_margin: float = 1.0
    mmd_min_samples_per_band: int = 4
    mmd_sample_mode: str = "boundary_only"
    mmd_sigma_mode: str = "median_heuristic"
    mmd_sigma: float = 1.0
    mmd_boundary_mode: str = "raw_boundary"
    mmd_uncertainty_threshold: float = 0.15
    mmd_project_dim: int = 128
    mmd_project_dropout: float = 0.05
    tie_qwk_tolerance: float = 0.005
    device: str = "auto"


@dataclass
class RecipeRunResult:
    recipe_id: str
    best_epoch: int
    inner_val_qwk: float
    inner_val_mae: float
    inner_val_acc: float
    metrics: Dict[str, Any]


@dataclass
class SelectionResult:
    selected_recipe_id: str
    selected_recipe: Recipe
    prompt_type: str
    candidate_recipe_ids: List[str]
    candidate_results: List[RecipeRunResult]
    inner_val_qwk: float
    inner_val_mae: float
    inner_val_acc: float
    tie_break_rank: int
    rule_gate_reasons: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def limit_split(split: Dict[str, np.ndarray], limit: Optional[int]) -> Dict[str, np.ndarray]:
    if limit is None or limit <= 0:
        return split
    n = min(limit, len(split["y_true"]))
    out: Dict[str, Any] = {}
    for k, v in split.items():
        if k == "rows" and isinstance(v, list):
            out[k] = v[:n]
        elif hasattr(v, "__getitem__") and not isinstance(v, (str, bytes, dict)):
            out[k] = v[:n]
        else:
            out[k] = v
    return out


def limit_evidence(
    evidence: LoadedEvidence,
    *,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> LoadedEvidence:
    splits = {
        "train": limit_split(evidence.splits["train"], limit_train),
        "val": limit_split(evidence.splits["val"], limit_val),
        "test": limit_split(evidence.splits["test"], limit_test),
    }
    return LoadedEvidence(
        prompt_id=evidence.prompt_id,
        fold=evidence.fold,
        score_min=evidence.score_min,
        score_max=evidence.score_max,
        num_anchors=evidence.num_anchors,
        input_dim=evidence.input_dim,
        splits=splits,
        cache_path=evidence.cache_path,
        payload=evidence.payload,
    )


def split_outer_train_for_inner(evidence: LoadedEvidence) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Use existing train/val as inner train/val to avoid any test leakage."""
    return evidence.splits["train"], evidence.splits["val"]


def full_outer_train(evidence: LoadedEvidence) -> Dict[str, np.ndarray]:
    return concat_splits([evidence.splits["train"], evidence.splits["val"]])


def fit_standardizer(train_split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    z = np.asarray(train_split["z"], dtype=np.float32)
    mu = z[:, 1:].mean(axis=0, keepdims=True)
    sd = z[:, 1:].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}


def apply_standardizer(split: Dict[str, np.ndarray], scaler: Dict[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(split["z"], dtype=np.float32).copy()
    z[:, 1:] = (z[:, 1:] - scaler["mu"]) / scaler["sd"]
    return z


def make_loader(
    split: Dict[str, np.ndarray],
    scaler: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    z = apply_standardizer(split, scaler)
    y = np.asarray(split["y_true"], dtype=np.int64)
    r = np.asarray(split["r_emb"], dtype=np.float32)
    raw = np.asarray(split["y_raw"], dtype=np.float32)
    prompt = np.asarray(split.get("prompt_id", np.zeros_like(y)), dtype=np.int64)
    uncertainty = np.asarray(split.get("uncertainty_scalar", np.zeros_like(raw)), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(z),
        torch.from_numpy(y),
        torch.from_numpy(r),
        torch.from_numpy(raw),
        torch.from_numpy(prompt),
        torch.from_numpy(uncertainty),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_mmd_config(recipe: Recipe, train_config: SelectorTrainConfig) -> Optional[MMDConfig]:
    if not recipe.mmd_enable:
        return None
    return MMDConfig(
        enable=True,
        feature_space="remb",
        scope="prompt_wise",
        band_mode="adjacent_only",
        sample_mode=train_config.mmd_sample_mode,
        kernel="rbf",
        sigma_mode=train_config.mmd_sigma_mode,
        sigma=train_config.mmd_sigma,
        margin=train_config.mmd_margin,
        lambda_sep=recipe.lambda_sep,
        warmup_epochs=train_config.mmd_warmup_epochs,
        min_samples_per_band=train_config.mmd_min_samples_per_band,
        boundary_mode=train_config.mmd_boundary_mode,
        uncertainty_threshold=train_config.mmd_uncertainty_threshold,
        raw_boundary_epsilon=recipe.mmd_raw_boundary_epsilon,
        num_bands=recipe.mmd_num_bands,
        project_dim=train_config.mmd_project_dim,
        project_dropout=train_config.mmd_project_dropout,
    )


def decode_for_recipe(
    outputs: Dict[str, torch.Tensor],
    z: torch.Tensor,
    recipe: Recipe,
    score_min: int,
    score_max: int,
) -> np.ndarray:
    raw_score = raw_scores_from_feature(z, score_min=score_min, score_max=score_max)
    max_delta = None
    if recipe.decode_mode == "blend_round":
        max_delta = recipe.max_delta_frac * float(score_max - score_min)
    decoded = decode_scores(
        raw_score=raw_score,
        expected_score=outputs["expected_score"],
        cum_probs=outputs["cum_probs"],
        score_min=score_min,
        score_max=score_max,
        decode_mode=recipe.decode_mode,
        blend_alpha=recipe.blend_alpha,
        max_delta=max_delta,
    )
    return decoded.detach().cpu().numpy().astype(int)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        qwk = 0.0
    else:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        if not math.isfinite(qwk):
            qwk = 0.0
    return {
        "qwk": qwk,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate_model(
    model: CoralOrdinalCalibrator,
    split: Dict[str, np.ndarray],
    scaler: Dict[str, np.ndarray],
    recipe: Recipe,
    score_min: int,
    score_max: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    z_np = apply_standardizer(split, scaler)
    z = torch.from_numpy(z_np).to(device)
    y_true = np.asarray(split["y_true"], dtype=np.int64)
    outputs = model(z)
    y_pred = decode_for_recipe(outputs, z, recipe, score_min, score_max)
    metrics = _metric_dict(y_true, y_pred)
    metrics["y_pred"] = y_pred
    return metrics


def train_recipe(
    recipe: Recipe,
    train_split: Dict[str, np.ndarray],
    val_split: Dict[str, np.ndarray],
    evidence: LoadedEvidence,
    train_config: SelectorTrainConfig,
    *,
    epochs: Optional[int] = None,
    seed_offset: int = 0,
    select_best: bool = True,
) -> Tuple[CoralOrdinalCalibrator, Dict[str, np.ndarray], RecipeRunResult]:
    epochs = epochs or train_config.epochs
    seed = train_config.seed + seed_offset
    set_seed(seed)
    device = get_device(train_config.device)
    scaler = fit_standardizer(train_split)
    loader = make_loader(train_split, scaler, train_config.batch_size, shuffle=True)
    input_dim = int(np.asarray(train_split["z"]).shape[1])
    model = CoralOrdinalCalibrator(
        CalibratorConfig(
            input_dim=input_dim,
            score_min=evidence.score_min,
            score_max=evidence.score_max,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
            lambda_qwk=recipe.lambda_qwk,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=train_config.weight_decay)
    mmd_cfg = build_mmd_config(recipe, train_config)
    mmd_projector = None
    if mmd_cfg and train_config.mmd_project_dim > 0:
        mmd_projector = MMDProjectionHead(
            input_dim=int(np.asarray(train_split["r_emb"]).shape[1]),
            output_dim=train_config.mmd_project_dim,
            dropout=train_config.mmd_project_dropout,
        ).to(device)
        optimizer.add_param_group({"params": mmd_projector.parameters()})

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_projector_state: Optional[Dict[str, torch.Tensor]] = None
    best_result: Optional[RecipeRunResult] = None

    for epoch in range(1, epochs + 1):
        model.train()
        if mmd_projector is not None:
            mmd_projector.train()
        use_mmd = mmd_cfg is not None and epoch > train_config.mmd_warmup_epochs
        epoch_loss: List[float] = []
        epoch_sep: List[float] = []

        for z, y, r_emb, raw, prompt, uncertainty in loader:
            z = z.to(device)
            y = y.to(device)
            r_emb = r_emb.to(device)
            raw = raw.to(device)
            prompt = prompt.to(device)
            uncertainty = uncertainty.to(device)
            loss, loss_outputs = calibrator_loss(model, z, y)
            sep_stats: Dict[str, Any] = {"loss_sep": 0.0}
            if use_mmd and mmd_cfg is not None:
                sep_features = mmd_projector(r_emb) if mmd_projector is not None else r_emb
                sep_loss, sep_stats = boundary_aware_mmd_separation_loss(
                    features=sep_features,
                    prompt_ids=prompt,
                    y_true=y,
                    y_raw=raw,
                    uncertainty=uncertainty,
                    score_min=evidence.score_min,
                    score_max=evidence.score_max,
                    config=mmd_cfg,
                )
                loss = loss + float(mmd_cfg.lambda_sep) * sep_loss
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if mmd_projector is not None:
                torch.nn.utils.clip_grad_norm_(mmd_projector.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_sep.append(float(sep_stats.get("loss_sep", 0.0)))

        val_metrics = evaluate_model(model, val_split, scaler, recipe, evidence.score_min, evidence.score_max, device)
        candidate = RecipeRunResult(
            recipe_id=recipe.recipe_id,
            best_epoch=epoch,
            inner_val_qwk=float(val_metrics["qwk"]),
            inner_val_mae=float(val_metrics["mae"]),
            inner_val_acc=float(val_metrics["acc"]),
            metrics={
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_loss)) if epoch_loss else float("nan"),
                "train_loss_sep": float(np.mean(epoch_sep)) if epoch_sep else 0.0,
                "val_qwk": float(val_metrics["qwk"]),
                "val_mae": float(val_metrics["mae"]),
                "val_acc": float(val_metrics["acc"]),
            },
        )
        if (not select_best) or best_result is None or better_candidate(
            candidate,
            best_result,
            recipe,
            recipe,
            qwk_tolerance=train_config.tie_qwk_tolerance,
        ):
            best_result = candidate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if mmd_projector is not None:
                best_projector_state = {k: v.detach().cpu().clone() for k, v in mmd_projector.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if mmd_projector is not None and best_projector_state is not None:
        mmd_projector.load_state_dict(best_projector_state)
    assert best_result is not None
    return model, scaler, best_result


def better_candidate(
    lhs: RecipeRunResult,
    rhs: RecipeRunResult,
    lhs_recipe: Recipe,
    rhs_recipe: Recipe,
    *,
    qwk_tolerance: float = 0.0,
) -> bool:
    qwk_delta = lhs.inner_val_qwk - rhs.inner_val_qwk
    if abs(qwk_delta) > float(qwk_tolerance):
        return qwk_delta > 0.0
    lhs_key = (
        -lhs.inner_val_mae,
        lhs.inner_val_acc,
        -lhs_recipe.simplicity_rank,
    )
    rhs_key = (
        -rhs.inner_val_mae,
        rhs.inner_val_acc,
        -rhs_recipe.simplicity_rank,
    )
    return lhs_key > rhs_key


def select_recipe(
    evidence: LoadedEvidence,
    recipes: Dict[str, Recipe],
    candidate_recipe_ids: Sequence[str],
    train_config: SelectorTrainConfig,
) -> SelectionResult:
    inner_train, inner_val = split_outer_train_for_inner(evidence)
    results: List[RecipeRunResult] = []
    best: Optional[RecipeRunResult] = None
    best_recipe: Optional[Recipe] = None
    for idx, recipe_id in enumerate(candidate_recipe_ids):
        recipe = recipes[recipe_id]
        _, _, result = train_recipe(
            recipe,
            inner_train,
            inner_val,
            evidence,
            train_config,
            seed_offset=evidence.prompt_id * 1000 + evidence.fold * 100 + idx,
        )
        results.append(result)
        if best is None or better_candidate(
            result,
            best,
            recipe,
            best_recipe,  # type: ignore[arg-type]
            qwk_tolerance=train_config.tie_qwk_tolerance,
        ):
            best = result
            best_recipe = recipe

    assert best is not None and best_recipe is not None
    return SelectionResult(
        selected_recipe_id=best.recipe_id,
        selected_recipe=best_recipe,
        prompt_type="",
        candidate_recipe_ids=list(candidate_recipe_ids),
        candidate_results=results,
        inner_val_qwk=best.inner_val_qwk,
        inner_val_mae=best.inner_val_mae,
        inner_val_acc=best.inner_val_acc,
        tie_break_rank=best_recipe.simplicity_rank,
        rule_gate_reasons="",
    )


def retrain_and_evaluate(
    evidence: LoadedEvidence,
    recipe: Recipe,
    train_config: SelectorTrainConfig,
    selected_epochs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    full_train = full_outer_train(evidence)
    test_split = evidence.splits["test"]
    model, scaler, _ = train_recipe(
        recipe,
        full_train,
        full_train,
        evidence,
        train_config,
        epochs=max(1, selected_epochs),
        seed_offset=evidence.prompt_id * 1000 + evidence.fold * 100 + 77,
        select_best=False,
    )
    device = get_device(train_config.device)
    metrics = evaluate_model(model, test_split, scaler, recipe, evidence.score_min, evidence.score_max, device)
    y_pred = metrics.pop("y_pred")
    y_true = np.asarray(test_split["y_true"], dtype=np.int64)
    y_raw = np.asarray(test_split["y_raw"], dtype=np.int64)
    raw_metrics = _metric_dict(y_true, y_raw)
    rows: List[Dict[str, Any]] = []
    essay_ids = np.asarray(test_split.get("essay_ids", np.arange(len(y_true))))
    for i in range(len(y_true)):
        rows.append(
            {
                "prompt": evidence.prompt_id,
                "fold": evidence.fold,
                "split": "test",
                "essay_id": int(essay_ids[i]),
                "y_true": int(y_true[i]),
                "y_raw": int(y_raw[i]),
                "y_pred": int(y_pred[i]),
                "recipe_id": recipe.recipe_id,
            }
        )
    metrics.update(
        {
            "local_wise_qwk": raw_metrics["qwk"],
            "local_wise_mae": raw_metrics["mae"],
            "local_wise_acc": raw_metrics["acc"],
        }
    )
    return metrics, rows


def recipe_run_to_dict(result: RecipeRunResult) -> Dict[str, Any]:
    data = asdict(result)
    return data


def run_prompt_fold(
    evidence_path: Path,
    out_dir: Path,
    *,
    mode: str = "pars",
    recipe_library: str = "v1",
    global_recipe: str = "R1",
    train_config: Optional[SelectorTrainConfig] = None,
    rule_config: Optional[RuleGateConfig] = None,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    train_config = train_config or SelectorTrainConfig()
    rule_config = rule_config or RuleGateConfig()
    evidence = load_evidence_cache(evidence_path)
    evidence = limit_evidence(
        evidence,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
    )
    recipes = get_recipe_library(recipe_library)
    features = compute_prompt_diagnostics(evidence)

    if mode == "pars":
        gate = apply_rule_gate(features, recipes, rule_config)
        candidate_ids = gate["candidate_recipes"]
        prompt_type = gate["prompt_type"]
        rule_reasons = gate["rule_gate_reasons"]
    elif mode == "manual":
        recipe_id = manual_prompt_recipe_v1(evidence.prompt_id)
        candidate_ids = [recipe_id]
        prompt_type = "manual"
        rule_reasons = "manual_prompt_recipe_v1"
        gate = {
            "prompt_type": prompt_type,
            "candidate_recipes": candidate_ids,
            "candidate_recipe_str": ",".join(candidate_ids),
            "mmd_gate_kept": recipes[recipe_id].mmd_enable,
            "rule_gate_reasons": rule_reasons,
        }
    elif mode == "global":
        candidate_ids = [global_recipe]
        prompt_type = "global"
        rule_reasons = f"global_recipe={global_recipe}"
        gate = {
            "prompt_type": prompt_type,
            "candidate_recipes": candidate_ids,
            "candidate_recipe_str": ",".join(candidate_ids),
            "mmd_gate_kept": recipes[candidate_ids[0]].mmd_enable,
            "rule_gate_reasons": rule_reasons,
        }
    else:
        raise ValueError(f"Unknown selector mode: {mode}")

    features.update({k: v for k, v in gate.items() if k != "candidate_recipes"})
    features["candidate_recipes"] = ",".join(candidate_ids)

    selection = select_recipe(evidence, recipes, candidate_ids, train_config)
    selection.prompt_type = prompt_type
    selection.rule_gate_reasons = rule_reasons
    selected_epochs = max(1, max((r.best_epoch for r in selection.candidate_results if r.recipe_id == selection.selected_recipe_id), default=train_config.epochs))
    final_metrics, per_essay = retrain_and_evaluate(evidence, selection.selected_recipe, train_config, selected_epochs)

    run_dir = out_dir / "runs" / f"p{evidence.prompt_id}_fold{evidence.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "per_essay.csv", per_essay)
    with (run_dir / "candidate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump([recipe_run_to_dict(r) for r in selection.candidate_results], f, indent=2, ensure_ascii=False)

    decision = {
        "prompt": evidence.prompt_id,
        "fold": evidence.fold,
        "mode": mode,
        "prompt_type": prompt_type,
        "candidate_recipes": ",".join(candidate_ids),
        "selected_recipe": selection.selected_recipe_id,
        "inner_val_qwk": selection.inner_val_qwk,
        "inner_val_mae": selection.inner_val_mae,
        "inner_val_acc": selection.inner_val_acc,
        "tie_break_rank": selection.tie_break_rank,
        "selected_epochs": selected_epochs,
        "candidate_metrics": json.dumps([recipe_run_to_dict(r) for r in selection.candidate_results], ensure_ascii=False),
        "rule_gate_reasons": rule_reasons,
    }

    main = {
        "prompt": evidence.prompt_id,
        "fold": evidence.fold,
        "mode": mode,
        "selected_recipe": selection.selected_recipe_id,
        "local_wise_qwk": final_metrics["local_wise_qwk"],
        "local_wise_mae": final_metrics["local_wise_mae"],
        "local_wise_acc": final_metrics["local_wise_acc"],
        "auto_pace_qwk": final_metrics["qwk"],
        "auto_pace_mae": final_metrics["mae"],
        "auto_pace_acc": final_metrics["acc"],
        "delta_qwk": final_metrics["qwk"] - final_metrics["local_wise_qwk"],
        "delta_mae": final_metrics["mae"] - final_metrics["local_wise_mae"],
        "delta_acc": final_metrics["acc"] - final_metrics["local_wise_acc"],
        "source_evidence": str(evidence.cache_path),
    }
    return features, decision, main


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(
    features_rows: Sequence[Dict[str, Any]],
    decision_rows: Sequence[Dict[str, Any]],
    main_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    recipe_counts: Dict[str, int] = {}
    prompt_type_counts: Dict[str, int] = {}
    for row in decision_rows:
        recipe_counts[str(row["selected_recipe"])] = recipe_counts.get(str(row["selected_recipe"]), 0) + 1
        prompt_type_counts[str(row["prompt_type"])] = prompt_type_counts.get(str(row["prompt_type"]), 0) + 1

    def mean_col(col: str) -> Optional[float]:
        vals = [float(r[col]) for r in main_rows if col in r and math.isfinite(float(r[col]))]
        return float(np.mean(vals)) if vals else None

    return {
        "n_runs": len(main_rows),
        "selected_recipe_counts": recipe_counts,
        "prompt_type_counts": prompt_type_counts,
        "mean_local_wise_qwk": mean_col("local_wise_qwk"),
        "mean_auto_pace_qwk": mean_col("auto_pace_qwk"),
        "mean_delta_qwk": mean_col("delta_qwk"),
        "mean_delta_mae": mean_col("delta_mae"),
        "mean_delta_acc": mean_col("delta_acc"),
        "feature_rows": len(features_rows),
        "decision_rows": len(decision_rows),
    }
