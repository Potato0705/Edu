"""Dual LLM backend for PACE-AES bridge layer.

Two backends share a common ``ScoringRequest`` / ``ScoringResult`` contract:

* :class:`OpenRouterBackend` — thin wrapper around ``wise_aes.call_llm`` so we
  reuse the exact API path that produced Layer-1 logs. No hidden states.
* :class:`LocalLlamaBackend` — local ``transformers`` inference for
  ``meta-llama/Meta-Llama-3.1-8B-Instruct`` (or a compatible path). Produces
  ``y_raw`` **and** ``h(x)`` (last-layer hidden state at the final generated
  token position). Minimum-viable version: single layer, single pool.

Prompt construction mirrors :data:`wise_aes.PromptIndividual.SCORING_TEMPLATE`
verbatim, so both backends see identical input text. For sanity / anchor
probing we skip dynamic RAG (``dynamic_ex="(None)"``) to keep prompts
deterministic; downstream feature extraction can re-enable RAG later.

Anchor caching
--------------
``h(e_k)`` for the anchors is expensive and stable per-fold, so we
cache it to disk:

    cache/pace_anchor_cache/{dataset}_p{prompt}_fold{f}.pt

The cache file is a ``torch.save``'d dict::

    {
        "anchor_ids": [...],
        "anchor_scores": [...],
        "hidden": Tensor[num_anchors, hidden_dim] (float16 / float32 depending on backend),
        "model_path": str,
        "backend_signature": str,   # hash of (model_path, dtype, pool)
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Lazy torch / transformers import so OpenRouter-only code paths work on boxes
# without GPU.
try:  # pragma: no cover - exercised on the GPU server
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

# Repo-root import so we can reuse wise_aes without editing it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_llm_token_limits(config: Dict, default: int = 768) -> Dict[str, int]:
    """Resolve per-call generation token budgets with backward compatibility."""
    llm_cfg = dict(config.get("llm", {}) or {})
    base = int(llm_cfg.get("max_new_tokens", llm_cfg.get("max_new_tokens_default", default)))
    return {
        "default": int(llm_cfg.get("max_new_tokens_default", base)),
        "scoring": int(llm_cfg.get("max_new_tokens_scoring", base)),
        "reflection": int(llm_cfg.get("max_new_tokens_reflection", base)),
        "induction": int(llm_cfg.get("max_new_tokens_induction", base)),
    }


# ---------------------------------------------------------------------------
# 1. Prompt construction (shared by both backends)
# ---------------------------------------------------------------------------


def anchor_role_name(idx: int, n_anchors: int) -> str:
    if n_anchors <= 3:
        roles = ["Low anchor", "Middle anchor", "High anchor"]
    else:
        roles = ["Low anchor", "Lower boundary anchor", "Upper boundary anchor", "High anchor"]
    return roles[idx] if idx < len(roles) else f"Additional anchor {idx + 1}"


def format_exemplars(exs: Sequence[Dict]) -> str:
    """Mirror :meth:`wise_aes.PromptIndividual._format_list` exactly."""
    if not exs:
        return "(None)"
    return "\n\n".join(
        [
            f"### {anchor_role_name(idx, len(exs))} (Known Score: {ex['domain1_score']})\n"
            f"Use this as a global scoring-scale reference, not as a retrieved neighbor.\n"
            f"Essay: {ex['essay_text'][:400]}..."
            for idx, ex in enumerate(exs)
        ]
    )


def build_scoring_prompt(
    *,
    instruction: str,
    static_exemplars: Sequence[Dict],
    essay_text: str,
    score_min: int,
    score_max: int,
    dynamic_ex: str = "(None)",
) -> str:
    """Construct the text that goes to either backend.

    Uses :data:`wise_aes.PromptIndividual.SCORING_TEMPLATE`. The ``dynamic_ex``
    slot is a plain string so the caller decides whether to skip RAG (default)
    or splice a pre-formatted RAG block.
    """
    from wise_aes import PromptIndividual  # local import: heavy deps

    return PromptIndividual.SCORING_TEMPLATE.format(
        instruction=instruction,
        static_ex=format_exemplars(static_exemplars),
        dynamic_ex=dynamic_ex,
        essay=essay_text,
        score_min=score_min,
        score_max=score_max,
    )


def build_representation_prompt(
    *,
    instruction: str,
    static_exemplars: Sequence[Dict],
    essay_text: str,
    score_min: int,
    score_max: int,
    known_score: Optional[int] = None,
    representation_target: Optional[str] = None,
) -> str:
    """Build the shared scoring-context prompt used to encode essays.

    Target essays and anchor essays are encoded in the same instruction,
    score-range, and reference-anchor context. Downstream code mean-pools the
    last-layer states over the ``essay_text`` span after ``Essay to Represent``.
    """
    anchor_blocks = []
    for idx, ex in enumerate(static_exemplars):
        anchor_blocks.append(
            f"Role: {anchor_role_name(idx, len(static_exemplars))}\n"
            f"Score: {ex['domain1_score']}\n"
            f"Essay: {ex['essay_text'][:400]}"
        )
    anchors_text = "\n\n".join(anchor_blocks) if anchor_blocks else "(None)"
    known_score_block = f"\n\nKnown Score:\n{known_score}" if known_score is not None else ""
    target_text = representation_target or "Encode the essay to be scored."
    return (
        "Scoring Instruction:\n"
        f"{instruction}\n\n"
        "Score Range:\n"
        f"{score_min}-{score_max}\n\n"
        "Reference Anchors:\n"
        f"{anchors_text}\n\n"
        "Essay to Represent:\n"
        f"{essay_text}"
        f"{known_score_block}\n\n"
        "Representation Target:\n"
        f"{target_text}"
    )


# ---------------------------------------------------------------------------
# 2. Contracts
# ---------------------------------------------------------------------------


@dataclass
class ScoringRequest:
    essay_id: int | str
    essay_text: str
    instruction: str
    static_exemplars: List[Dict]
    score_min: int
    score_max: int
    dynamic_ex: str = "(None)"


@dataclass
class ScoringResult:
    essay_id: int | str
    y_raw: int
    raw_text: str                    # full model output (for JSON parse debugging)
    prompt_text: str
    hidden: Optional["torch.Tensor"] = None  # shape (hidden_dim,), cpu float
    wallclock_sec: float = 0.0
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. OpenRouter backend (reuses wise_aes.call_llm)
# ---------------------------------------------------------------------------


class OpenRouterBackend:
    """Score via the same OpenRouter path that produced Layer-1 logs.

    Does **not** return hidden states. Used by the sanity check to compare
    ``y_raw`` distributions against the local HF backend.
    """

    def __init__(self, config: Dict):
        self.config = config
        self._ensure_exp_manager_mock()

    def _ensure_exp_manager_mock(self) -> None:
        import wise_aes  # local import

        if wise_aes.EXP_MANAGER is not None:
            return

        class _MockExpManager:
            def __init__(self, cfg):
                self.config = cfg

            def log_llm_trace(self, _):  # noqa: D401
                pass

            def track_usage(self, *_):
                pass

            def count_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

        wise_aes.EXP_MANAGER = _MockExpManager(self.config)

    def score(self, req: ScoringRequest) -> ScoringResult:
        from wise_aes import PromptIndividual, call_llm

        prompt = build_scoring_prompt(
            instruction=req.instruction,
            static_exemplars=req.static_exemplars,
            essay_text=req.essay_text,
            score_min=req.score_min,
            score_max=req.score_max,
            dynamic_ex=req.dynamic_ex,
        )
        t0 = time.time()
        response = call_llm(
            prompt,
            temperature=self.config["llm"].get("temperature_scoring", 0.0),
            call_type="pace_sanity_openrouter",
        )
        wall = time.time() - t0

        ind = PromptIndividual.__new__(PromptIndividual)
        ind.config = self.config
        parsed = ind._extract_json_safe(response)
        y_raw = _parse_score(parsed, response, req.score_min, req.score_max)
        return ScoringResult(
            essay_id=req.essay_id,
            y_raw=y_raw,
            raw_text=response,
            prompt_text=prompt,
            hidden=None,
            wallclock_sec=wall,
            meta={"backend": "openrouter"},
        )


# ---------------------------------------------------------------------------
# 4. Local Llama backend (HF transformers, bf16, output_hidden_states=True)
# ---------------------------------------------------------------------------


class LocalLlamaBackend:
    """Local HF inference with hidden-state capture.

    Minimum-viable version (Phase 2):
    * Only the last layer's hidden state is captured.
    * Only the **final generated token position** is used as h(x).
      This position corresponds to the state that emitted the final token
      of the model's answer — after it has committed to a score.
    * Greedy decoding at T=0 for determinism.

    Later phases can add alternative pooling (mean over the JSON span,
    mid-layer taps, etc.) — this module intentionally exposes a single
    pool for now.
    """

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __init__(
        self,
        config: Dict,
        *,
        model_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 768,
        pool: str = "final_gen_token",
        load_in_4bit: bool = False,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError(
                "torch is required for LocalLlamaBackend; install torch+transformers."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.config = config
        self.model_path = model_path or self.DEFAULT_MODEL
        self.dtype_str = dtype
        self.device = device
        token_limits = resolve_llm_token_limits(config, default=max_new_tokens)
        self.max_new_tokens = int(token_limits["default"])
        self.max_new_tokens_scoring = int(token_limits["scoring"])
        self.max_new_tokens_reflection = int(token_limits["reflection"])
        self.max_new_tokens_induction = int(token_limits["induction"])
        self.load_in_4bit = load_in_4bit
        if pool != "final_gen_token":
            raise NotImplementedError(
                f"Only pool='final_gen_token' is supported in the MVP. Got {pool!r}."
            )
        self.pool = pool

        torch_dtype = getattr(torch, dtype)
        print(f"[LocalLlama] Loading tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            print(
                f"[LocalLlama] Loading model 4-bit NF4 (compute_dtype={dtype}) ..."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=qcfg,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        else:
            print(
                f"[LocalLlama] Loading model (dtype={dtype}, device_map={device}) ..."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size
        print(
            f"[LocalLlama] Ready. hidden_size={self.hidden_dim}, "
            f"model_max_len={self.tokenizer.model_max_length}"
        )
        self._lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage = {
            "scoring_calls": 0,
            "generation_calls": 0,
            "representation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "representation_tokens": 0,
        }

    # ---- public API -----------------------------------------------------

    def signature(self) -> str:
        """Stable hash of (model, dtype, pool, quant). Written into anchor cache."""
        quant = "4bit_nf4" if self.load_in_4bit else "none"
        blob = (
            f"{self.model_path}|{self.dtype_str}|{self.pool}|{quant}|"
            f"max_new_tokens={self.max_new_tokens}|"
            f"score={self.max_new_tokens_scoring}|"
            f"reflect={self.max_new_tokens_reflection}|"
            f"induce={self.max_new_tokens_induction}"
        ).encode("utf-8")
        return hashlib.md5(blob).hexdigest()

    def score(self, req: ScoringRequest) -> ScoringResult:
        from wise_aes import PromptIndividual

        prompt_text = build_scoring_prompt(
            instruction=req.instruction,
            static_exemplars=req.static_exemplars,
            essay_text=req.essay_text,
            score_min=req.score_min,
            score_max=req.score_max,
            dynamic_ex=req.dynamic_ex,
        )
        chat_prompt = self._apply_chat_template(prompt_text)
        prompt_tokens = self._count_tokens(chat_prompt)
        t0 = time.time()
        with self._lock:
            response_text, final_hidden = self._generate(
                chat_prompt,
                max_new_tokens=self.max_new_tokens_scoring,
            )
        wall = time.time() - t0
        completion_tokens = self._count_tokens(response_text)
        self._record_usage(
            scoring_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        ind = PromptIndividual.__new__(PromptIndividual)
        ind.config = self.config
        parsed = ind._extract_json_safe(response_text)
        y_raw = _parse_score(parsed, response_text, req.score_min, req.score_max)
        return ScoringResult(
            essay_id=req.essay_id,
            y_raw=y_raw,
            raw_text=response_text,
            prompt_text=prompt_text,
            hidden=final_hidden.to("cpu").float(),  # cpu fp32 for stable downstream
            wallclock_sec=wall,
            meta={
                "backend": "local_llama",
                "model_path": self.model_path,
                "dtype": self.dtype_str,
                "pool": self.pool,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    def encode_scoring_context(
        self,
        *,
        instruction: str,
        static_exemplars: Sequence[Dict],
        essay_text: str,
        score_min: int,
        score_max: int,
        known_score: Optional[int] = None,
        representation_target: Optional[str] = None,
    ) -> "torch.Tensor":
        """Return mean-pooled last-layer states over the essay span.

        This is the hidden representation used by WISE-PACE evidence. It keeps
        target and anchor essays comparable by encoding both inside the same
        scoring-context template, then pooling only tokens from the essay placed
        after ``Essay to Represent``.
        """
        prompt_text = build_representation_prompt(
            instruction=instruction,
            static_exemplars=static_exemplars,
            essay_text=essay_text,
            score_min=score_min,
            score_max=score_max,
            known_score=known_score,
            representation_target=representation_target,
        )
        chat_prompt = self._apply_chat_template(prompt_text)
        representation_tokens = self._count_tokens(chat_prompt)
        marker = "Essay to Represent:\n"
        marker_pos = chat_prompt.rfind(marker)
        if marker_pos < 0:
            raise RuntimeError("Representation prompt marker not found.")
        essay_start = chat_prompt.find(essay_text, marker_pos + len(marker))
        if essay_start < 0:
            raise RuntimeError("Essay span not found in representation prompt.")
        essay_end = essay_start + len(essay_text)

        with self._lock:
            hidden = self._encode_span_mean(chat_prompt, essay_start, essay_end)
        self._record_usage(
            representation_calls=1,
            representation_tokens=representation_tokens,
        )
        return hidden.to("cpu").float()

    def encode_text_mean(self, text: str) -> "torch.Tensor":
        """Return a mean-pooled last-layer representation for free text."""
        text = str(text or "").strip()
        if not text:
            return torch.zeros(self.hidden_dim, dtype=torch.float32)
        representation_tokens = self._count_tokens(text)
        with self._lock:
            hidden = self._encode_span_mean(text, 0, len(text))
        self._record_usage(
            representation_calls=1,
            representation_tokens=representation_tokens,
        )
        return hidden.to("cpu").float()

    def usage_snapshot(self) -> Dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def usage_delta(self, before: Dict[str, int]) -> Dict[str, int]:
        now = self.usage_snapshot()
        return {k: int(now.get(k, 0) - before.get(k, 0)) for k in now}

    def record_generation_usage(self, prompt_text: str, response_text: str) -> None:
        self._record_usage(
            generation_calls=1,
            prompt_tokens=self._count_tokens(prompt_text),
            completion_tokens=self._count_tokens(response_text),
        )

    # ---- internals ------------------------------------------------------

    def _apply_chat_template(self, user_text: str) -> str:
        """Llama-3.1 chat template. The scoring prompt is the only user turn."""
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return max(1, len(text) // 4)

    def _record_usage(self, **kwargs: int) -> None:
        with self._usage_lock:
            for key, value in kwargs.items():
                self._usage[key] = self._usage.get(key, 0) + int(value)

    def _generate(
        self,
        chat_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, "torch.Tensor"]:
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        generation_tokens = int(max_new_tokens or self.max_new_tokens)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=generation_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # out.hidden_states: tuple with len == n_generated_tokens.
        # Each element is a tuple over layers; element 0 covers the prompt
        # (batch, prompt_len, hidden); elements [1..] each cover one new token
        # (batch, 1, hidden).
        if len(out.hidden_states) == 0:
            raise RuntimeError("Generator produced no hidden states.")
        last_step = out.hidden_states[-1]
        last_layer_hidden = last_step[-1]
        # Shape (batch=1, seq=1 for last step, hidden)
        final_hidden = last_layer_hidden[0, -1, :].detach()
        return response_text, final_hidden

    def _encode_span_mean(
        self,
        chat_prompt: str,
        char_start: int,
        char_end: int,
    ) -> "torch.Tensor":
        encoded = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0]
        inputs = encoded.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1][0]

        mask = []
        for start, end in offsets.tolist():
            mask.append(end > char_start and start < char_end)
        mask_t = torch.tensor(mask, device=last_hidden.device, dtype=torch.bool)
        if not bool(mask_t.any()):
            raise RuntimeError("No tokens overlapped requested essay span.")
        return last_hidden[mask_t].mean(dim=0).detach()


# ---------------------------------------------------------------------------
# 5. Anchor hidden-state cache
# ---------------------------------------------------------------------------


@dataclass
class AnchorRecord:
    essay_id: int | str
    domain1_score: int
    band: str
    essay_text: str


class AnchorHiddenCache:
    """Offline cache for anchor hidden states per (dataset, prompt, fold).

    Usage::

        cache = AnchorHiddenCache(cache_root="cache/pace_anchor_cache")
        entry = cache.load_or_build(
            dataset="asap", prompt=1, fold=0,
            anchors=[...],
            backend=local_backend,
            instruction=I_star, score_min=2, score_max=12,
        )
        anchor_hidden = entry.hidden    # (num_anchors, hidden_dim) on cpu fp32
    """

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset: str, prompt: int, fold: int) -> Path:
        return self.cache_root / f"{dataset}_p{prompt}_fold{fold}.pt"

    def load_or_build(
        self,
        *,
        dataset: str,
        prompt: int,
        fold: int,
        anchors: Sequence[AnchorRecord],
        backend: "LocalLlamaBackend",
        instruction: str,
        score_min: int,
        score_max: int,
    ) -> "AnchorCacheEntry":
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch required for anchor cache.")
        path = self.path_for(dataset, prompt, fold)
        sig = backend.signature()
        expected_bands = [a.band for a in anchors]
        if path.exists():
            try:
                entry_dict = torch.load(path, map_location="cpu")
                if (
                    entry_dict.get("backend_signature") == sig
                    and len(entry_dict.get("anchor_ids", [])) == len(anchors)
                    and list(entry_dict.get("bands", [])) == expected_bands
                ):
                    return AnchorCacheEntry(**entry_dict)
                else:
                    print(
                        f"[AnchorCache] Cache mismatch at {path.name}: "
                        "rebuilding."
                    )
            except Exception as exc:
                print(f"[AnchorCache] Failed to load {path}: {exc}. Rebuilding.")

        if len(anchors) < 3:
            raise ValueError(
                f"Expected at least 3 anchors; got {len(anchors)}."
            )
        ids: List = []
        scores: List[int] = []
        bands: List[str] = []
        hiddens: List["torch.Tensor"] = []
        for a in anchors:
            req = ScoringRequest(
                essay_id=a.essay_id,
                essay_text=a.essay_text,
                instruction=instruction,
                static_exemplars=[],  # anchors scored without their own peers
                score_min=score_min,
                score_max=score_max,
                dynamic_ex="(None)",
            )
            res = backend.score(req)
            if res.hidden is None:
                raise RuntimeError("Backend did not return hidden state.")
            ids.append(a.essay_id)
            scores.append(a.domain1_score)
            bands.append(a.band)
            hiddens.append(res.hidden)
            print(
                f"[AnchorCache] {dataset} P{prompt} fold{fold} {a.band} "
                f"id={a.essay_id} y_raw={res.y_raw} wall={res.wallclock_sec:.2f}s"
            )
        hidden_tensor = torch.stack(hiddens, dim=0)
        entry = AnchorCacheEntry(
            anchor_ids=ids,
            anchor_scores=scores,
            bands=bands,
            hidden=hidden_tensor,
            model_path=backend.model_path,
            backend_signature=sig,
        )
        torch.save(entry.__dict__, path)
        print(f"[AnchorCache] Saved {path}")
        return entry


@dataclass
class AnchorCacheEntry:
    anchor_ids: List
    anchor_scores: List[int]
    bands: List[str]
    hidden: "torch.Tensor"
    model_path: str
    backend_signature: str


# ---------------------------------------------------------------------------
# 6. Score parsing helpers (shared)
# ---------------------------------------------------------------------------


_RE_INT = None


def _parse_score(
    parsed: Optional[Dict], raw_text: str, score_min: int, score_max: int
) -> int:
    if parsed is not None and "final_score" in parsed:
        try:
            score = int(parsed["final_score"])
            if score_min <= score <= score_max:
                return score
        except Exception:
            pass
    return _parse_score_from_text(raw_text, score_min, score_max)


def _parse_score_from_text(raw_text: str, score_min: int, score_max: int) -> int:
    """Regex fallback matching wise_aes.py:564-572."""
    import re

    global _RE_INT
    if _RE_INT is None:
        _RE_INT = re.compile(r"-?\d+")

    # 1) Try JSON first (generous)
    try:
        import json as _json

        for open_brace_pos in range(raw_text.rfind("}"), -1, -1):
            if open_brace_pos < 0:
                break
            if raw_text[open_brace_pos] == "{":
                try:
                    obj = _json.loads(raw_text[open_brace_pos : raw_text.rfind("}") + 1])
                    if isinstance(obj, dict) and "final_score" in obj:
                        score = int(obj["final_score"])
                        if score_min <= score <= score_max:
                            return score
                except Exception:
                    continue
    except Exception:
        pass

    # 2) Look for explicit score mentions before falling back to raw integers.
    explicit_patterns = [
        re.compile(r'"final_score"\s*:\s*(-?\d+)', re.IGNORECASE),
        re.compile(
            r"\bfinal score\b(?:\s*(?:is|of|=|:))?\s*(-?\d+)", re.IGNORECASE
        ),
        re.compile(
            r"\bi would assign\b.*?\bfinal score\b(?:\s*(?:is|of|=|:))?\s*(-?\d+)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bassign(?:ed|ing)?\b.*?\bfinal score\b(?:\s*(?:is|of|=|:))?\s*(-?\d+)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bscore of\b\s*(-?\d+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in explicit_patterns:
        matches = pattern.findall(raw_text)
        for num_str in reversed(matches):
            try:
                val = int(num_str)
            except ValueError:
                continue
            if score_min <= val <= score_max:
                return val

    # 3) Reverse integer scan only inside the tail after a likely scoring cue.
    lowered = raw_text.lower()
    cue_positions = [
        lowered.rfind('"final_score"'),
        lowered.rfind("final score"),
        lowered.rfind("i would assign"),
        lowered.rfind("therefore"),
        lowered.rfind("score of"),
    ]
    start = max(cue_positions)
    if start >= 0:
        matches = _RE_INT.findall(raw_text[start:])
        for num_str in reversed(matches):
            try:
                val = int(num_str)
            except ValueError:
                continue
            if score_min <= val <= score_max:
                return val

    # 4) Median fallback
    return (score_min + score_max) // 2


# ---------------------------------------------------------------------------
# 7. Convenience: load Layer-1 champion (I*, E*) from a logs/exp_* dir
# ---------------------------------------------------------------------------


def load_layer1_champion(
    exp_dir: str | Path, gen: int = 25
) -> Dict:
    """Return {'instruction', 'static_exemplar_ids', 'config'} for the fold champion."""
    import yaml  # local import

    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen_path = exp_dir / "generations" / f"gen_{gen:03d}.json"
    if not gen_path.exists():
        available = sorted(exp_dir.glob("generations/gen_*.json"))
        gen_path = available[-1] if available else None
        if gen_path is None:
            raise FileNotFoundError(f"No generations under {exp_dir}")
    with gen_path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    champion = max(snap["population"], key=lambda p: p.get("fitness", -1.0))
    return {
        "config": cfg,
        "instruction": champion["full_instruction"],
        "static_exemplar_ids": champion["static_exemplar_ids"],
        "gen_path": str(gen_path),
        "val_fitness": champion.get("fitness"),
    }


__all__ = [
    "ScoringRequest",
    "ScoringResult",
    "OpenRouterBackend",
    "LocalLlamaBackend",
    "AnchorRecord",
    "AnchorCacheEntry",
    "AnchorHiddenCache",
    "build_scoring_prompt",
    "build_representation_prompt",
    "format_exemplars",
    "load_layer1_champion",
]
