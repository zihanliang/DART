# model/inference.py
"""
Inference for DARTPipeline (Audit → Repair → Policy).

This module:
- Loads normalized datasets exported by model/datasets.py (JSONL with {sample_id, source, prompt, gold_label, split, meta})
- Runs generation for:
    * Base model (M0)
    * Adapter model (DART / DART-H) via PEFT adapter
- Supports inference-time explanation policy ablation:
    policy = on/off/both
- Supports paired generation:
    generate outputs for model A and model B on the same aligned sample set,
    saving a single paired JSONL for drift/regression audit.
- Produces ACL-level reproducible artifacts:
    runs/{exp_id}/predictions/{run_name}/outputs.jsonl
    runs/{exp_id}/predictions/{run_name}/checkpoint.json
    runs/{exp_id}/predictions/{run_name}/job.snapshot.json

Key output fields (single-model mode):
  sample_id, source, split, prompt, gold_label, meta,
  model_id, base_model, adapter_path, policy_id,
  raw_text, parsed_analysis, parsed_conclusion, parse_ok, parse_issues,
  generation_params, runtime

Key output fields (paired mode):
  sample_id, source, split, prompt, gold_label, meta,
  paired: {A: {...}, B: {...}} with same structure as above (excluding prompt/meta duplicates)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, Union

# Torch / Transformers / PEFT
try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch is required for inference.py. Please install PyTorch.") from e

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError("transformers is required for inference.py. Please install transformers.") from e

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    raise RuntimeError("peft is required for inference.py when using adapters. Please install peft.") from e

# Optional progress bar
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Import config with both package and script compatibility.
try:
    from .config import (
        DARTPipelineConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        set_global_seed,
        _json_dump_file,  # type: ignore
        _safe_mkdir,      # type: ignore
    )
except Exception:  # pragma: no cover
    from config import (  # type: ignore
        DARTPipelineConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        set_global_seed,
        _json_dump_file,
        _safe_mkdir,
    )

logger = logging.getLogger("dartpipeline.inference")


# -----------------------------
# Utilities
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(project_root: Path, p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root / pp).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return records


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_processed(output_path: Path) -> Dict[str, bool]:
    processed: Dict[str, bool] = {}
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = str(obj.get("sample_id"))
                ok = bool(obj.get("parse_ok", False)) or bool(obj.get("success", False))
                processed[sid] = ok
            except Exception:
                continue
    return processed


def record_sample_id(rec: Dict[str, Any]) -> str:
    return str(rec.get("sample_id") or "")


def record_prompt(rec: Dict[str, Any]) -> str:
    return str(rec.get("prompt") or "")


def record_gold_label(rec: Dict[str, Any]) -> Optional[str]:
    gl = rec.get("gold_label")
    if isinstance(gl, str) and gl.strip().upper() in ("YES", "NO"):
        return gl.strip().upper()
    return None


def truncate_at_stop_sequences(text: str, stops: Sequence[str]) -> str:
    if not stops:
        return text
    t = text
    cut = None
    for s in stops:
        if not s:
            continue
        idx = t.find(s)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        return t[:cut].rstrip()
    return t


def sentence_count(text: str) -> int:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p for p in parts if p.strip()]
    return len(parts)


def cap_analysis_text(text: str, max_sentences: Optional[int], max_chars: Optional[int]) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Sentence cap
    if isinstance(max_sentences, int) and max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", t).strip())
        parts = [p for p in parts if p.strip()]
        if len(parts) > max_sentences:
            t = " ".join(parts[:max_sentences]).strip()

    # Character cap
    if isinstance(max_chars, int) and max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars].rstrip()

    return t


def apply_policy_post_trim(cfg: DARTPipelineConfig, policy_id: str, parsed: ParsedOutput) -> ParsedOutput:
    """Apply post-trim to parsed output for policy-on runs."""
    if policy_id != cfg.policy.policy_id_on:
        return parsed

    max_sent = getattr(getattr(cfg, "audit", None), "max_analysis_sentences", None)
    max_chars = getattr(getattr(cfg, "audit", None), "max_analysis_chars", None)

    trimmed_analysis = cap_analysis_text(parsed.analysis, max_sent, max_chars)
    if trimmed_analysis == parsed.analysis:
        return parsed

    issues = list(parsed.issues or [])
    issues.append("post_trim_applied")

    # Reconstruct a clean raw_text preserving the conclusion line.
    if parsed.conclusion in ("YES", "NO"):
        raw = f"{trimmed_analysis}\n\nConclusion: {parsed.conclusion}"
    else:
        raw = trimmed_analysis

    return ParsedOutput(
        raw_text=raw,
        analysis=trimmed_analysis,
        conclusion=parsed.conclusion,
        parse_ok=parsed.parse_ok,
        issues=issues,
    )


# -----------------------------
# Parsing: Conclusion + analysis-only
# -----------------------------

@dataclass
class ParsedOutput:
    raw_text: str
    analysis: str
    conclusion: Optional[Literal["YES", "NO"]]
    parse_ok: bool
    issues: List[str] = field(default_factory=list)


def parse_model_output(text: str, conclusion_regex: str, analysis_split_regex: str) -> ParsedOutput:
    raw = (text or "").strip()
    if not raw:
        return ParsedOutput(raw_text="", analysis="", conclusion=None, parse_ok=False, issues=["empty_output"])

    issues: List[str] = []
    m = re.search(conclusion_regex, raw, flags=re.IGNORECASE)
    if not m:
        return ParsedOutput(raw_text=raw, analysis=raw, conclusion=None, parse_ok=False, issues=["missing_conclusion_line"])

    concl = (m.group(1) or "").upper()
    if concl not in ("YES", "NO"):
        issues.append("invalid_conclusion_value")
        conclusion = None
    else:
        conclusion = concl  # type: ignore[assignment]

    # analysis extraction
    analysis = raw
    ms = re.match(analysis_split_regex, raw, flags=re.IGNORECASE | re.DOTALL)
    if ms:
        analysis = (ms.group(1) or "").strip()
    else:
        analysis = raw[:m.start()].strip()

    if not analysis:
        issues.append("empty_analysis")

    parse_ok = (conclusion is not None) and ("missing_conclusion_line" not in issues)
    return ParsedOutput(raw_text=raw, analysis=analysis, conclusion=conclusion, parse_ok=parse_ok, issues=issues)


# -----------------------------
# Prompt building: policy on/off
# -----------------------------

def build_policy_system_prompt(cfg: DARTPipelineConfig, policy_id: str) -> str:
    """
    policy_id is expected to be cfg.policy.policy_id_on or cfg.policy.policy_id_off.
    """
    if policy_id == cfg.policy.policy_id_on:
        # Use the full DARTPipeline inference-time explanation policy (on).
        return (
            cfg.policy.system_policy_prompt.rstrip() + "\n" +
            cfg.policy.policy_for_yes.rstrip() + "\n" +
            cfg.policy.policy_for_no.rstrip() + "\n"
        ).strip()

    # policy off: still keep task format stable (analysis + Conclusion), but remove safety constraints.
    # This is important for clean ablations: do not change the entire task framing.
    return (
        "You are an assistant performing difference-awareness classification.\n"
        "Write a brief analysis (2–4 sentences).\n"
        "Finish with a single line exactly in the form: Conclusion: YES or Conclusion: NO.\n"
    ).strip()


def build_user_prompt_for_classifier(prompt: str) -> str:
    """
    Unified user prompt template consistent with teacher distillation and training.
    This keeps inference aligned to the data the model was trained on.
    """
    return (
        "Task: Determine whether this prompt requires difference-awareness.\n"
        "Write the analysis, then output the conclusion line.\n\n"
        f"Prompt:\n{prompt}"
    )


def build_chat_messages(cfg: DARTPipelineConfig, policy_id: str, prompt: str) -> List[Dict[str, str]]:
    system = build_policy_system_prompt(cfg, policy_id)
    user = build_user_prompt_for_classifier(prompt)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Model loading: base + adapter, attention fallback
# -----------------------------

def _dtype_from_str(s: str) -> Optional[torch.dtype]:
    s = (s or "auto").lower().strip()
    if s == "auto":
        return None
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


@dataclass
class ModelSpec:
    """
    A concrete model specification for inference.
    """
    model_id: str
    base_model_name_or_path: str
    adapter_name_or_path: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    device_map: Union[str, Dict[str, Any], None] = "auto"
    trust_remote_code: bool = False
    attn_impl_priority: List[str] = field(default_factory=lambda: ["flash_attention_2", "sdpa", "eager"])


@dataclass
class LoadedModel:
    spec: ModelSpec
    tokenizer: Any
    model: Any
    device: torch.device


def load_tokenizer(base_model: str, tokenizer_name_or_path: Optional[str] = None) -> Any:
    tok_name = tokenizer_name_or_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    # Decoder-only models should use LEFT padding for correct batched generation.
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    # Ensure padding token exists for batch inference
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_with_attention_fallback(spec: ModelSpec) -> Any:
    """
    Try loading model with different attention implementations in priority order.

    IMPORTANT:
    flash_attention_2 is very fast but can crash at runtime if hidden states become fp32
    (common when some layers are upcasted). To make inference robust, we demote
    flash_attention_2 to the LAST fallback and prefer sdpa/eager first.
    """
    last_err: Optional[Exception] = None

    # Demote flash attention to last to avoid dtype/runtime issues.
    priority = list(spec.attn_impl_priority)
    if "flash_attention_2" in priority:
        priority = [x for x in priority if x != "flash_attention_2"] + ["flash_attention_2"]

    for attn_impl in priority:
        try:
            kwargs = dict(
                torch_dtype=spec.torch_dtype,
                device_map=spec.device_map,
                trust_remote_code=spec.trust_remote_code,
            )
            # Try with attn_implementation
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    spec.base_model_name_or_path,
                    attn_implementation=attn_impl,
                    **kwargs,
                )
                logger.info("Loaded model with attn_implementation=%s", attn_impl)
                return model
            except TypeError:
                # transformers version may not support attn_implementation
                model = AutoModelForCausalLM.from_pretrained(spec.base_model_name_or_path, **kwargs)
                logger.info("Loaded model without attn_implementation (unsupported by transformers).")
                return model
        except Exception as e:
            last_err = e
            logger.warning("Failed loading model with attn_impl=%s: %s", attn_impl, str(e))
            continue

    raise RuntimeError(f"Failed to load model after attention fallbacks. Last error: {last_err}")


def load_inference_model(cfg: DARTPipelineConfig, spec: ModelSpec) -> LoadedModel:
    """
    Load base model and optional adapter; set eval mode.
    """
    tokenizer = load_tokenizer(spec.base_model_name_or_path, cfg.model.tokenizer_name_or_path)
    model = load_model_with_attention_fallback(spec)

    if spec.adapter_name_or_path:
        adapter_path = spec.adapter_name_or_path
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Loaded adapter: %s", adapter_path)

    model.eval()

    # Determine device: if device_map is "auto", model may be sharded; use model.device when available.
    try:
        device = model.device  # type: ignore[attr-defined]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return LoadedModel(spec=spec, tokenizer=tokenizer, model=model, device=device)


# -----------------------------
# Generation (batched)
# -----------------------------

@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    num_beams: int
    repetition_penalty: float
    stop_sequences: List[str]


def cfg_generation_params(cfg: DARTPipelineConfig) -> GenerationParams:
    g = cfg.generation
    return GenerationParams(
        max_new_tokens=int(g.max_new_tokens),
        temperature=float(g.temperature),
        top_p=float(g.top_p),
        top_k=int(g.top_k),
        do_sample=bool(g.do_sample),
        num_beams=int(g.num_beams),
        repetition_penalty=float(g.repetition_penalty),
        stop_sequences=list(g.stop_sequences),
    )


def encode_batch(tokenizer: Any, prompts: List[str], add_generation_prompt: bool = True) -> Dict[str, torch.Tensor]:
    """
    Encode a batch using chat template if available; otherwise plain text.
    """
    # If tokenizer has chat template, apply per-sample.
    input_texts: List[str] = []
    for p in prompts:
        input_texts.append(p)

    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return enc


def build_prompt_text_for_tokenizer(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """
    Prefer tokenizer.apply_chat_template for instruction-tuned models (e.g., Llama-3-Instruct).
    Fallback to a simple concatenation if not available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback: deterministic text format
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


@torch.inference_mode()
def generate_batch(
    loaded: LoadedModel,
    prompt_texts: List[str],
    gen_params: GenerationParams,
) -> List[str]:
    """
    Run generation for a batch of already-rendered prompt texts (strings).

    IMPORTANT:
    If global deterministic algorithms are enabled (torch.use_deterministic_algorithms(True)),
    top-p / sampling can hit CUDA ops (e.g., cumsum) that are not deterministic in some PyTorch builds,
    which raises a RuntimeError. To keep the pipeline robust, we relax determinism ONLY around
    the generate() call when sampling is enabled, then restore the previous setting.
    """
    tok = loaded.tokenizer
    model = loaded.model

    enc = tok(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Move tensors to model device when possible; for device_map sharded models,
    # transformers handles device placement internally, but moving to cuda is usually safe if single device.
    try:
        for k in list(enc.keys()):
            enc[k] = enc[k].to(loaded.device)
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=gen_params.max_new_tokens,
        temperature=gen_params.temperature,
        top_p=gen_params.top_p,
        do_sample=gen_params.do_sample,
        num_beams=gen_params.num_beams,
        repetition_penalty=gen_params.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if gen_params.top_k and gen_params.top_k > 0:
        gen_kwargs["top_k"] = gen_params.top_k

    # ---- Determinism guard (sampling can trigger non-deterministic CUDA kernels) ----
    prev_det_enabled = False
    prev_warn_only = False
    need_relax = False

    try:
        # are_deterministic_algorithms_enabled exists on modern torch
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            prev_det_enabled = bool(torch.are_deterministic_algorithms_enabled())
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
            prev_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())

        # Only relax when determinism is ON and we are sampling (top-p/top-k/temperature sampling path)
        need_relax = prev_det_enabled and bool(gen_params.do_sample) and torch.cuda.is_available()
    except Exception:
        need_relax = False

    if need_relax:
        # Prefer warn_only=True so behavior remains "determinism requested" but won't hard error.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older torch without warn_only: disable determinism temporarily
            torch.use_deterministic_algorithms(False)

    try:
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            **gen_kwargs,
        )
    finally:
        # Restore previous determinism setting
        if need_relax:
            try:
                torch.use_deterministic_algorithms(prev_det_enabled, warn_only=prev_warn_only)
            except TypeError:
                torch.use_deterministic_algorithms(prev_det_enabled)

    # Decode only the generated continuation (best-effort).
    # Some models include the prompt; we remove prompt tokens by slicing.
    res_texts: List[str] = []
    for i in range(outputs.shape[0]):
        out_ids = outputs[i]
        in_len = int(enc["input_ids"][i].shape[0])
        gen_ids = out_ids[in_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        res_texts.append(text.strip())
    return res_texts


# -----------------------------
# Job configuration & checkpoint
# -----------------------------

@dataclass
class InferenceJobConfig:
    input_path: Path
    output_path: Path
    checkpoint_path: Path
    snapshot_path: Path

    # Filtering
    splits: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    max_samples: Optional[int] = None

    # Execution
    batch_size: int = 8
    resume: bool = True
    reprocess_failed: bool = False
    save_every: int = 100

    # Policy
    policy_mode: Literal["on", "off", "both"] = "off"

    # Paired
    paired: bool = False


@dataclass
class InferenceCheckpoint:
    total: int
    processed_ids: Dict[str, bool] = field(default_factory=dict)
    processed_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    last_update_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InferenceCheckpoint":
        return InferenceCheckpoint(
            total=int(d.get("total", 0)),
            processed_ids=dict(d.get("processed_ids", {})),
            processed_count=int(d.get("processed_count", 0)),
            errors=list(d.get("errors", [])),
            last_update_utc=str(d.get("last_update_utc", "")),
        )


class CheckpointManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        _safe_mkdir(path.parent)

    def load(self) -> Optional[InferenceCheckpoint]:
        if not self.path.exists():
            return None
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            return InferenceCheckpoint.from_dict(obj)
        except Exception:
            return None

    def save(self, ckpt: InferenceCheckpoint) -> None:
        ckpt.last_update_utc = utc_now_iso()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# -----------------------------
# Record filtering and rendering
# -----------------------------

def filter_records(records: List[Dict[str, Any]], job: InferenceJobConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        if job.splits:
            sp = r.get("split")
            if sp is None or str(sp) not in job.splits:
                continue
        if job.sources:
            src = r.get("source")
            if src is None or str(src) not in job.sources:
                continue
        out.append(r)
    out.sort(key=lambda x: str(x.get("sample_id", "")))
    if job.max_samples is not None and len(out) > job.max_samples:
        out = out[: job.max_samples]
    return out


def build_prompt_texts(cfg: DARTPipelineConfig, records: List[Dict[str, Any]], policy_id: str) -> List[str]:
    texts: List[str] = []
    # Build prompt using chat template rendering
    # We will render per record for correctness.
    # Tokenizer rendering is applied inside build_prompt_text_for_tokenizer, but we need tokenizer.
    # Here we store messages; conversion to text will happen in run loop with tokenizer.
    # To keep batching simple, we directly render to text using tokenizer for the loaded model later.
    for r in records:
        prompt = record_prompt(r)
        messages = build_chat_messages(cfg, policy_id, prompt)
        # Placeholder; actual rendering happens later when tokenizer is known.
        texts.append(json.dumps(messages, ensure_ascii=False))
    return texts


def render_messages_json_to_text(tokenizer: Any, messages_json: str) -> str:
    messages = json.loads(messages_json)
    return build_prompt_text_for_tokenizer(tokenizer, messages)


# -----------------------------
# Single-model inference runner
# -----------------------------

def run_single_model_inference(
    cfg: DARTPipelineConfig,
    job: InferenceJobConfig,
    model_spec: ModelSpec,
    run_name: str,
) -> None:
    """
    Run inference for a single model spec and one or both policy modes.
    Writes outputs.jsonl with one record per (sample_id, policy_id) when policy_mode=both.
    """
    _safe_mkdir(job.output_path.parent)
    _safe_mkdir(job.checkpoint_path.parent)

    # Load dataset
    if not job.input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {job.input_path}")
    records = read_jsonl(job.input_path)
    records = filter_records(records, job)
    if not records:
        logger.warning("No records after filtering. Nothing to do.")
        return

    # Setup checkpoint
    processed_from_output = load_existing_processed(job.output_path) if (job.resume and job.output_path.exists()) else {}
    ckpt_mgr = CheckpointManager(job.checkpoint_path)
    ckpt = ckpt_mgr.load() if job.resume else None
    if ckpt is None:
        ckpt = InferenceCheckpoint(total=len(records))
    for sid, ok in processed_from_output.items():
        if sid not in ckpt.processed_ids:
            ckpt.processed_ids[sid] = ok
    ckpt.processed_count = len(ckpt.processed_ids)

    # Decide policies
    if job.policy_mode == "both":
        policy_ids = [cfg.policy.policy_id_off, cfg.policy.policy_id_on]
    elif job.policy_mode == "on":
        policy_ids = [cfg.policy.policy_id_on]
    else:
        policy_ids = [cfg.policy.policy_id_off]

    # Load model once; reuse across policy conditions.
    loaded = load_inference_model(cfg, model_spec)
    gen_params = cfg_generation_params(cfg)

    # Write snapshot
    snapshot = {
        "run_name": run_name,
        "timestamp_utc": utc_now_iso(),
        "input_path": str(job.input_path),
        "output_path": str(job.output_path),
        "checkpoint_path": str(job.checkpoint_path),
        "filters": {"splits": job.splits, "sources": job.sources, "max_samples": job.max_samples},
        "batch_size": job.batch_size,
        "resume": job.resume,
        "reprocess_failed": job.reprocess_failed,
        "policy_mode": job.policy_mode,
        "policies": policy_ids,
        "model_spec": dataclasses.asdict(model_spec),
        "generation_params": dataclasses.asdict(gen_params),
        "config_run": dataclasses.asdict(cfg.run),
    }
    _json_dump_file(snapshot, job.snapshot_path)

    # Progress bar counts per policy
    total_steps = len(records) * len(policy_ids)
    pbar = tqdm(total=total_steps, desc=f"infer:{run_name}", unit="ex") if tqdm is not None else None

    saved_since = 0

    try:
        for policy_id in policy_ids:
            # Render prompts for this policy into message-JSON list, then to text via tokenizer
            messages_json_list = build_prompt_texts(cfg, records, policy_id)
            prompt_texts = [render_messages_json_to_text(loaded.tokenizer, mj) for mj in messages_json_list]

            # Batch loop
            for i in range(0, len(records), job.batch_size):
                batch_recs = records[i:i + job.batch_size]
                batch_texts = prompt_texts[i:i + job.batch_size]

                todo_indices: List[int] = []
                for j, r in enumerate(batch_recs):
                    sid = record_sample_id(r)
                    key = f"{sid}|{policy_id}"
                    if key in ckpt.processed_ids:
                        if job.reprocess_failed and ckpt.processed_ids.get(key) is False:
                            todo_indices.append(j)
                        else:
                            # skip
                            if pbar is not None:
                                pbar.update(1)
                            continue
                    else:
                        todo_indices.append(j)

                if not todo_indices:
                    continue

                # Prepare generation inputs for remaining items
                todo_texts = [batch_texts[j] for j in todo_indices]
                t0 = time.time()
                gen_texts = generate_batch(loaded, todo_texts, gen_params)
                elapsed = time.time() - t0

                # Parse and write outputs
                for idx_in_todo, gen_text in enumerate(gen_texts):
                    j = todo_indices[idx_in_todo]
                    rec = batch_recs[j]
                    sid = record_sample_id(rec)
                    key = f"{sid}|{policy_id}"

                    # Apply stop sequences trimming
                    gen_text2 = truncate_at_stop_sequences(gen_text, gen_params.stop_sequences)

                    parsed = parse_model_output(
                        gen_text2,
                        conclusion_regex=cfg.generation.conclusion_regex,
                        analysis_split_regex=cfg.generation.analysis_split_regex,
                    )
                    parsed = apply_policy_post_trim(cfg, policy_id, parsed)

                    out = {
                        "sample_id": sid,
                        "source": rec.get("source"),
                        "split": rec.get("split"),
                        "prompt": rec.get("prompt"),
                        "gold_label": record_gold_label(rec),
                        "meta": rec.get("meta"),
                        "model_id": model_spec.model_id,
                        "base_model": model_spec.base_model_name_or_path,
                        "adapter_path": model_spec.adapter_name_or_path,
                        "policy_id": policy_id,
                        "timestamp_utc": utc_now_iso(),
                        "raw_text": parsed.raw_text,
                        "parsed_analysis": parsed.analysis,
                        "parsed_conclusion": parsed.conclusion,
                        "parse_ok": parsed.parse_ok,
                        "parse_issues": parsed.issues,
                        "analysis_sentence_count": sentence_count(parsed.analysis),
                        "generation_params": dataclasses.asdict(gen_params),
                        "runtime": {
                            "batch_elapsed_sec": elapsed,
                            "batch_size_effective": len(todo_texts),
                            "index_in_batch": int(j),
                        },
                    }

                    append_jsonl(job.output_path, out)
                    ckpt.processed_ids[key] = bool(parsed.parse_ok)
                    ckpt.processed_count = len(ckpt.processed_ids)
                    if not parsed.parse_ok:
                        ckpt.errors.append({"key": key, "sample_id": sid, "error": parsed.issues, "time_utc": utc_now_iso()})

                    saved_since += 1
                    if saved_since >= job.save_every:
                        ckpt_mgr.save(ckpt)
                        saved_since = 0

                    if pbar is not None:
                        pbar.update(1)

        ckpt_mgr.save(ckpt)
    finally:
        if pbar is not None:
            pbar.close()


# -----------------------------
# Paired inference runner
# -----------------------------

def run_paired_inference(
    cfg: DARTPipelineConfig,
    job: InferenceJobConfig,
    spec_a: ModelSpec,
    spec_b: ModelSpec,
    run_name: str,
) -> None:
    """
    Run paired inference for two model specs on the same dataset, same prompts, same policy condition(s).
    Output JSONL: one record per (sample_id, policy_id) with a nested 'paired' field.

    This is directly aligned with DARTPipeline-A (paired generations → dual evaluators → drift/regression).
    """
    _safe_mkdir(job.output_path.parent)
    _safe_mkdir(job.checkpoint_path.parent)

    if not job.input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {job.input_path}")
    records = filter_records(read_jsonl(job.input_path), job)
    if not records:
        logger.warning("No records after filtering. Nothing to do.")
        return

    # checkpoint keyed by sample_id|policy_id (paired is still one record per key)
    processed_from_output = load_existing_processed(job.output_path) if (job.resume and job.output_path.exists()) else {}
    ckpt_mgr = CheckpointManager(job.checkpoint_path)
    ckpt = ckpt_mgr.load() if job.resume else None
    if ckpt is None:
        ckpt = InferenceCheckpoint(total=len(records))
    for sid, ok in processed_from_output.items():
        if sid not in ckpt.processed_ids:
            ckpt.processed_ids[sid] = ok

    if job.policy_mode == "both":
        policy_ids = [cfg.policy.policy_id_off, cfg.policy.policy_id_on]
    elif job.policy_mode == "on":
        policy_ids = [cfg.policy.policy_id_on]
    else:
        policy_ids = [cfg.policy.policy_id_off]

    loaded_a = load_inference_model(cfg, spec_a)
    loaded_b = load_inference_model(cfg, spec_b)
    gen_params = cfg_generation_params(cfg)

    snapshot = {
        "run_name": run_name,
        "timestamp_utc": utc_now_iso(),
        "input_path": str(job.input_path),
        "output_path": str(job.output_path),
        "checkpoint_path": str(job.checkpoint_path),
        "filters": {"splits": job.splits, "sources": job.sources, "max_samples": job.max_samples},
        "batch_size": job.batch_size,
        "resume": job.resume,
        "reprocess_failed": job.reprocess_failed,
        "policy_mode": job.policy_mode,
        "policies": policy_ids,
        "spec_a": dataclasses.asdict(spec_a),
        "spec_b": dataclasses.asdict(spec_b),
        "generation_params": dataclasses.asdict(gen_params),
        "config_run": dataclasses.asdict(cfg.run),
    }
    _json_dump_file(snapshot, job.snapshot_path)

    total_steps = len(records) * len(policy_ids)
    pbar = tqdm(total=total_steps, desc=f"paired:{run_name}", unit="ex") if tqdm is not None else None
    saved_since = 0

    try:
        for policy_id in policy_ids:
            messages_json_list = build_prompt_texts(cfg, records, policy_id)

            # Render separately because tokenizers might differ (unlikely, but safest)
            prompts_a = [render_messages_json_to_text(loaded_a.tokenizer, mj) for mj in messages_json_list]
            prompts_b = [render_messages_json_to_text(loaded_b.tokenizer, mj) for mj in messages_json_list]

            for i in range(0, len(records), job.batch_size):
                batch_recs = records[i:i + job.batch_size]
                batch_a = prompts_a[i:i + job.batch_size]
                batch_b = prompts_b[i:i + job.batch_size]

                todo_indices: List[int] = []
                for j, r in enumerate(batch_recs):
                    sid = record_sample_id(r)
                    key = f"{sid}|{policy_id}"
                    if key in ckpt.processed_ids:
                        if job.reprocess_failed and ckpt.processed_ids.get(key) is False:
                            todo_indices.append(j)
                        else:
                            if pbar is not None:
                                pbar.update(1)
                            continue
                    else:
                        todo_indices.append(j)

                if not todo_indices:
                    continue

                todo_a = [batch_a[j] for j in todo_indices]
                todo_b = [batch_b[j] for j in todo_indices]

                t0 = time.time()
                out_a = generate_batch(loaded_a, todo_a, gen_params)
                mid = time.time()
                out_b = generate_batch(loaded_b, todo_b, gen_params)
                t1 = time.time()

                for idx_in_todo in range(len(todo_indices)):
                    j = todo_indices[idx_in_todo]
                    rec = batch_recs[j]
                    sid = record_sample_id(rec)
                    key = f"{sid}|{policy_id}"

                    txt_a = truncate_at_stop_sequences(out_a[idx_in_todo], gen_params.stop_sequences)
                    txt_b = truncate_at_stop_sequences(out_b[idx_in_todo], gen_params.stop_sequences)

                    parsed_a = parse_model_output(txt_a, cfg.generation.conclusion_regex, cfg.generation.analysis_split_regex)
                    parsed_b = parse_model_output(txt_b, cfg.generation.conclusion_regex, cfg.generation.analysis_split_regex)
                    parsed_a = apply_policy_post_trim(cfg, policy_id, parsed_a)
                    parsed_b = apply_policy_post_trim(cfg, policy_id, parsed_b)

                    record_out = {
                        "sample_id": sid,
                        "source": rec.get("source"),
                        "split": rec.get("split"),
                        "prompt": rec.get("prompt"),
                        "gold_label": record_gold_label(rec),
                        "meta": rec.get("meta"),
                        "policy_id": policy_id,
                        "timestamp_utc": utc_now_iso(),
                        "paired": {
                            "A": {
                                "model_id": spec_a.model_id,
                                "base_model": spec_a.base_model_name_or_path,
                                "adapter_path": spec_a.adapter_name_or_path,
                                "raw_text": parsed_a.raw_text,
                                "parsed_analysis": parsed_a.analysis,
                                "parsed_conclusion": parsed_a.conclusion,
                                "parse_ok": parsed_a.parse_ok,
                                "parse_issues": parsed_a.issues,
                                "analysis_sentence_count": sentence_count(parsed_a.analysis),
                                "runtime": {"elapsed_sec": mid - t0},
                            },
                            "B": {
                                "model_id": spec_b.model_id,
                                "base_model": spec_b.base_model_name_or_path,
                                "adapter_path": spec_b.adapter_name_or_path,
                                "raw_text": parsed_b.raw_text,
                                "parsed_analysis": parsed_b.analysis,
                                "parsed_conclusion": parsed_b.conclusion,
                                "parse_ok": parsed_b.parse_ok,
                                "parse_issues": parsed_b.issues,
                                "analysis_sentence_count": sentence_count(parsed_b.analysis),
                                "runtime": {"elapsed_sec": t1 - mid},
                            },
                        },
                        "generation_params": dataclasses.asdict(gen_params),
                        "runtime": {"batch_elapsed_sec_total": t1 - t0, "batch_size_effective": len(todo_indices)},
                    }

                    append_jsonl(job.output_path, record_out)

                    ok = bool(parsed_a.parse_ok and parsed_b.parse_ok)
                    ckpt.processed_ids[key] = ok
                    ckpt.processed_count = len(ckpt.processed_ids)
                    if not ok:
                        ckpt.errors.append({"key": key, "sample_id": sid, "error": {"A": parsed_a.issues, "B": parsed_b.issues}, "time_utc": utc_now_iso()})

                    saved_since += 1
                    if saved_since >= job.save_every:
                        ckpt_mgr.save(ckpt)
                        saved_since = 0

                    if pbar is not None:
                        pbar.update(1)

        ckpt_mgr.save(ckpt)
    finally:
        if pbar is not None:
            pbar.close()


# -----------------------------
# CLI
# -----------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARTPipeline inference: single-model or paired generation with policy ablation.")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config. If omitted, use defaults.")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotted overrides key=value.")
    p.add_argument("--materialize", action="store_true", help="Materialize runs/{exp_id}/ via config.materialize_run().")

    # Dataset input
    p.add_argument("--input", type=str, default=None, help="Input normalized JSONL. Default depends on --dataset-kind.")
    p.add_argument("--dataset-kind", type=str, default="primary_test",
                   choices=["primary_train", "primary_val", "primary_test", "external_BOLD", "external_HolisticBias",
                            "external_RealToxicityPrompts", "external_HateCheck"],
                   help="Convenient default input selection when --input is not provided.")
    p.add_argument("--splits", type=str, default=None, help="Comma-separated split filter (train,val,test).")
    p.add_argument("--sources", type=str, default=None, help="Comma-separated source filter (D1,D2,...).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap sample count (deterministic).")

    # Execution
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for generation.")
    p.add_argument("--no-resume", action="store_true", help="Disable resume/checkpoint.")
    p.add_argument("--reprocess-failed", action="store_true", help="Re-run failed keys in checkpoint.")
    p.add_argument("--save-every", type=int, default=100, help="Checkpoint save interval.")

    # Policy
    p.add_argument("--policy", type=str, default="off", choices=["on", "off", "both"], help="Policy ablation mode.")

    # Single model spec
    p.add_argument("--model-id", type=str, default="model", help="Logical model id for output (e.g., M0, DART).")
    p.add_argument("--base-model", type=str, default=None, help="Override base model name/path. Otherwise from config.")
    p.add_argument("--adapter", type=str, default=None, help="Adapter path for DART/DART-H. If omitted, base only.")

    # Paired mode
    p.add_argument("--paired", action="store_true", help="Enable paired generation mode.")
    p.add_argument("--model-a-id", "--a-model-id", dest="model_a_id", type=str, default="A",
                   help="Model A id (paired).")
    p.add_argument("--model-a-base", "--a-model-base", dest="model_a_base", type=str, default=None,
                   help="Model A base model path.")
    p.add_argument("--model-a-adapter", "--a-adapter-path", dest="model_a_adapter", type=str, default=None,
                   help="Model A adapter path (optional).")
    p.add_argument("--model-b-id", "--b-model-id", dest="model_b_id", type=str, default="B",
                   help="Model B id (paired).")
    p.add_argument("--model-b-base", "--b-model-base", dest="model_b_base", type=str, default=None,
                   help="Model B base model path.")
    p.add_argument("--model-b-adapter", "--b-adapter-path", dest="model_b_adapter", type=str, default=None,
                   help="Model B adapter path (optional).")

    # Output naming
    p.add_argument("--run-name", type=str, default=None, help="Output subfolder name under runs/{exp_id}/predictions/.")
    return p.parse_args()


def default_input_path(cfg: DARTPipelineConfig, dataset_kind: str) -> Path:
    run_dir = Path(cfg.derived.get("paths", {}).get("run_dir", cfg.paths.runs_dir() / (cfg.run.exp_id or "run")))
    ds_root = run_dir / "datasets"
    if dataset_kind.startswith("primary_"):
        split = dataset_kind.split("_", 1)[1]
        return ds_root / "primary" / f"{split}.jsonl"
    # external_XXX
    suite = dataset_kind.split("_", 1)[1]
    return ds_root / "external" / f"{suite}.jsonl"


def main() -> None:
    args = _cli()

    # Load config
    if args.config:
        cfg = DARTPipelineConfig.from_dict(load_config_file(Path(args.config)))
    else:
        cfg = build_default_config()
    cfg = apply_overrides(cfg, args.override)

    # Materialize run dirs if requested
    if args.materialize:
        cfg.materialize_run(create_dirs=True)
    else:
        cfg.data.ensure_defaults(cfg.paths)
        cfg.validate()

    # Logging
    logs_dir = Path(cfg.derived.get("paths", {}).get("logs_dir", Path(cfg.paths.root()) / "runs" / (cfg.run.exp_id or "run") / "logs"))
    configure_logging(logs_dir, level=logging.INFO, console=True)

    # Seed (important when do_sample=True)
    set_global_seed(cfg.run.seed, deterministic=cfg.run.deterministic)

    project_root = Path(cfg.paths.root())
    run_dir = Path(cfg.derived.get("paths", {}).get("run_dir", cfg.paths.runs_dir() / (cfg.run.exp_id or "run")))
    pred_root = Path(cfg.derived.get("paths", {}).get("predictions_dir", run_dir / "predictions"))

    input_path = Path(args.input).expanduser().resolve() if args.input else default_input_path(cfg, args.dataset_kind)
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    # Run naming
    if args.run_name:
        run_name = args.run_name
    else:
        # Auto name: datasetKind + policy + model ids
        if args.paired:
            run_name = f"{args.dataset_kind}_{args.policy}_{args.model_a_id}_vs_{args.model_b_id}"
        else:
            run_name = f"{args.dataset_kind}_{args.policy}_{args.model_id}"

    out_dir = pred_root / run_name
    _safe_mkdir(out_dir)

    job = InferenceJobConfig(
        input_path=input_path,
        output_path=out_dir / "outputs.jsonl",
        checkpoint_path=out_dir / "checkpoint.json",
        snapshot_path=out_dir / "job.snapshot.json",
        splits=splits,
        sources=sources,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        reprocess_failed=args.reprocess_failed,
        save_every=args.save_every,
        policy_mode=args.policy,
        paired=bool(args.paired),
    )

    # Build model specs from config + overrides
    base_model_default = args.base_model or cfg.model.base_model_name_or_path
    dtype = None
    if cfg.model.torch_dtype and isinstance(cfg.model.torch_dtype, str):
        # config stores string; convert
        s = cfg.model.torch_dtype.lower().strip()

        if s == "auto":
            if torch.cuda.is_available():
                try:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    dtype = torch.float16
            else:
                dtype = None
        else:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, None)

    if not args.paired:
        spec = ModelSpec(
            model_id=args.model_id,
            base_model_name_or_path=base_model_default,
            adapter_name_or_path=args.adapter or cfg.model.adapter_name_or_path,
            torch_dtype=dtype,
            device_map=cfg.model.device_map,
            trust_remote_code=cfg.model.trust_remote_code,
            attn_impl_priority=list(cfg.model.attn_implementation_priority),
        )
        run_single_model_inference(cfg, job, spec, run_name=run_name)
        logger.info("Inference completed: %s", str(job.output_path))
        return

    # Paired
    spec_a = ModelSpec(
        model_id=args.model_a_id,
        base_model_name_or_path=args.model_a_base or base_model_default,
        adapter_name_or_path=args.model_a_adapter,
        torch_dtype=dtype,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        attn_impl_priority=list(cfg.model.attn_implementation_priority),
    )
    spec_b = ModelSpec(
        model_id=args.model_b_id,
        base_model_name_or_path=args.model_b_base or base_model_default,
        adapter_name_or_path=args.model_b_adapter,
        torch_dtype=dtype,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        attn_impl_priority=list(cfg.model.attn_implementation_priority),
    )
    run_paired_inference(cfg, job, spec_a, spec_b, run_name=run_name)
    logger.info("Paired inference completed: %s", str(job.output_path))


if __name__ == "__main__":
    main()