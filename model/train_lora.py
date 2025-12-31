# model/train_lora.py
"""
This module trains:
- DART: base model (M0) + LoRA, supervised on teacher distillation outputs (analysis + Conclusion: YES/NO)
- DART-H: continued SFT starting from an existing DART adapter, supervised on repair "safe analysis targets"
          (optionally mixed with original distill data; severity-weighted oversampling supported)

Inputs (expected, produced by model/teacher_generate.py):
- runs/{exp_id}/teacher_outputs/distill/outputs.jsonl
- runs/{exp_id}/teacher_outputs/repair/outputs.jsonl

Each teacher output line is expected to contain at least:
  sample_id, prompt, gold_label, raw_text, success, parse_ok/violations (best-effort)
We robustly filter/clean even if some fields are missing.

Outputs:
- runs/{exp_id}/adapters/{adapter_name}/
    - final/ (adapter weights + tokenizer)
    - checkpoints/ (trainer checkpoints, optional)
    - job.snapshot.json (full provenance)
    - data.manifest.json (input file metadata)
    - train.metrics.json / eval.metrics.json
    - training_summary.json

Design choices aligned with DARTPipeline plan:
- Training target is short analysis (2–4 sentences) + "Conclusion: YES/NO".
- Inference-time Policy (stage C) is NOT baked into training by default (clean ablation).
- Prompt tokens are masked in labels so we train only on the assistant continuation (teacher target).

Dependencies:
- torch
- transformers
- peft
Optional:
- bitsandbytes (for QLoRA)
- tqdm (progress)

Recommended usage:
  # Train DART from distill targets
  python -m model.train_lora --materialize --stage dart --adapter-name DART

  # Train DART-H from repair targets, initializing from DART adapter
  python -m model.train_lora --materialize --stage dart_h --adapter-name DART_H --init-adapter runs/<exp_id>/adapters/DART/final

  # Mixed repair recipe (distill + weighted repair)
  python -m model.train_lora --materialize --stage dart_h --recipe mix --adapter-name DART_H \
      --init-adapter runs/<exp_id>/adapters/DART/final --repair-oversample severe=3,extreme=4,moderate=2,mild=1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)

# Some PEFT versions expose this helper; we import conditionally.
try:  # pragma: no cover
    from peft import prepare_model_for_kbit_training  # type: ignore
except Exception:  # pragma: no cover
    prepare_model_for_kbit_training = None  # type: ignore

# Optional bitsandbytes config for QLoRA
try:  # pragma: no cover
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

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

logger = logging.getLogger("dartpipeline.train_lora")


# -----------------------------
# Time & IO helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_manifest(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return items


# -----------------------------
# DARTPipeline-aligned prompting (same as inference)
# -----------------------------

def build_policy_system_prompt(cfg: DARTPipelineConfig, policy_id: str) -> str:
    """
    policy_id is expected to be cfg.policy.policy_id_on or cfg.policy.policy_id_off.
    For training, default is policy off (clean ablation); safe style is learned from targets.
    """
    if policy_id == cfg.policy.policy_id_on:
        return (
            cfg.policy.system_policy_prompt.rstrip() + "\n" +
            cfg.policy.policy_for_yes.rstrip() + "\n" +
            cfg.policy.policy_for_no.rstrip() + "\n"
        ).strip()

    return (
        "You are an assistant performing difference-awareness classification.\n"
        "Write a brief analysis (2–4 sentences).\n"
        "Finish with a single line exactly in the form: Conclusion: YES or Conclusion: NO.\n"
    ).strip()


def build_user_prompt(prompt: str) -> str:
    return (
        "Task: Determine whether this prompt requires difference-awareness.\n"
        "Write the analysis, then output the conclusion line.\n\n"
        f"Prompt:\n{prompt}"
    )


def build_messages(cfg: DARTPipelineConfig, policy_id: str, prompt: str, assistant_target: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": build_policy_system_prompt(cfg, policy_id)},
        {"role": "user", "content": build_user_prompt(prompt)},
    ]
    if assistant_target is not None:
        msgs.append({"role": "assistant", "content": assistant_target})
    return msgs


def render_chat(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Use tokenizer.apply_chat_template when available; fallback to deterministic plain text.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass

    # Fallback text format (stable across environments)
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"[{role}]\n{content}")
    if add_generation_prompt:
        parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


# -----------------------------
# Teacher record cleaning & filtering
# -----------------------------

ConclusionLabel = Literal["YES", "NO"]


def _extract_conclusion(text: str) -> Optional[ConclusionLabel]:
    if not text:
        return None
    m = None
    try:
        m = next(re.finditer(r"(?mi)^\s*Conclusion\s*:\s*(YES|NO)\s*$", text))
    except Exception:
        m = None
    if m is None:
        # fallback: search anywhere
        m2 = re.search(r"(?i)Conclusion\s*:\s*(YES|NO)", text)
        if not m2:
            return None
        v = m2.group(1).upper()
        return "YES" if v == "YES" else "NO"
    v = m.group(1).upper()
    return "YES" if v == "YES" else "NO"


def _truncate_after_conclusion(text: str) -> str:
    """
    If there is trailing material after the conclusion line, drop it for training stability.
    """
    if not text:
        return ""
    # Find the first occurrence of a conclusion line; keep up to end of that line.
    m = re.search(r"(?mi)^\s*Conclusion\s*:\s*(YES|NO)\s*$", text)
    if not m:
        return text.strip()
    end = m.end()
    return text[:end].strip()


def _normalize_gold_label(rec: Dict[str, Any]) -> Optional[ConclusionLabel]:
    gl = rec.get("gold_label")
    if isinstance(gl, str):
        g = gl.strip().upper()
        if g in ("YES", "NO"):
            return g  # type: ignore[return-value]
        if g in ("DIFF", "EQUAL"):
            return "YES" if g == "DIFF" else "NO"
    return None


def _teacher_success_flag(rec: Dict[str, Any]) -> bool:
    s = rec.get("success")
    if isinstance(s, bool):
        return s
    # some pipelines only have parse_ok
    po = rec.get("parse_ok")
    if isinstance(po, bool):
        return po
    return False


@dataclass
class CleanTeacherExample:
    sample_id: str
    prompt: str
    target: str
    gold_label: Optional[ConclusionLabel]
    source: Optional[str] = None
    split: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    task: Optional[str] = None  # distill/repair
    severity_bin: Optional[str] = None  # mild/moderate/severe/extreme
    provenance: Dict[str, Any] = field(default_factory=dict)


def clean_teacher_outputs(
    jsonl_path: Path,
    *,
    require_success: bool = True,
    require_conclusion: bool = True,
    require_label_match: bool = True,
) -> List[CleanTeacherExample]:
    """
    Convert teacher_generate outputs.jsonl into clean SFT pairs.
    """
    raw = read_jsonl(jsonl_path)
    out: List[CleanTeacherExample] = []

    for rec in raw:
        sample_id = str(rec.get("sample_id") or "").strip()
        prompt = str(rec.get("prompt") or "").strip()
        raw_text = str(rec.get("raw_text") or rec.get("target") or "").strip()

        if not sample_id or not prompt or not raw_text:
            continue

        if require_success and not _teacher_success_flag(rec):
            continue

        target = _truncate_after_conclusion(raw_text)
        concl = _extract_conclusion(target)
        if require_conclusion and concl is None:
            continue

        gold = _normalize_gold_label(rec)

        # Some teacher outputs provide parsed_conclusion explicitly.
        parsed_concl = rec.get("parsed_conclusion")
        if isinstance(parsed_concl, str) and parsed_concl.strip().upper() in ("YES", "NO"):
            concl = parsed_concl.strip().upper()  # type: ignore[assignment]

        if require_label_match and gold is not None and concl is not None and concl != gold:
            # discard inconsistent teacher targets
            continue

        # Severity bin for repair oversampling (best-effort)
        sev = None
        meta = rec.get("meta")
        if isinstance(meta, dict):
            sev = meta.get("severity_bin") or meta.get("severity") or meta.get("regression_severity")
        if sev is None and isinstance(rec.get("regression"), dict):
            sev = rec["regression"].get("severity_bin")
        if sev is not None:
            sev = str(sev).strip().lower()

        out.append(
            CleanTeacherExample(
                sample_id=sample_id,
                prompt=prompt,
                target=target,
                gold_label=gold,
                source=str(rec.get("source")) if rec.get("source") is not None else None,
                split=str(rec.get("split")) if rec.get("split") is not None else None,
                meta=dict(meta) if isinstance(meta, dict) else {},
                task=str(rec.get("task")) if rec.get("task") is not None else None,
                severity_bin=sev,
                provenance={
                    "teacher_model": rec.get("teacher_model"),
                    "teacher_provider": rec.get("teacher_provider"),
                    "timestamp_utc": rec.get("timestamp_utc"),
                    "format_fix_rounds_used": rec.get("format_fix_rounds_used"),
                },
            )
        )

    return out


def dedupe_examples(examples: List[CleanTeacherExample]) -> List[CleanTeacherExample]:
    """
    Deduplicate by (sample_id, target_hash) to allow both distill and repair targets for same sample_id.
    """
    seen = set()
    out: List[CleanTeacherExample] = []
    for ex in examples:
        key = (ex.sample_id, hashlib.sha256(ex.target.encode("utf-8")).hexdigest()[:12])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def split_train_val(examples: List[CleanTeacherExample], val_ratio: float, seed: int) -> Tuple[List[CleanTeacherExample], List[CleanTeacherExample]]:
    if val_ratio <= 0.0:
        return examples, []
    rnd = random.Random(seed)
    idx = list(range(len(examples)))
    rnd.shuffle(idx)
    n_val = max(1, int(round(len(examples) * val_ratio)))
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, ex in enumerate(examples):
        (val if i in val_idx else train).append(ex)
    return train, val


def oversample_by_severity(
    repair_examples: List[CleanTeacherExample],
    severity_to_repeat: Dict[str, int],
) -> List[CleanTeacherExample]:
    """
    Duplicate repair examples based on severity bin; bins not found default to 1.
    """
    out: List[CleanTeacherExample] = []
    for ex in repair_examples:
        sev = (ex.severity_bin or "").lower().strip()
        rep = int(severity_to_repeat.get(sev, 1))
        rep = max(1, rep)
        for _ in range(rep):
            out.append(ex)
    return out


# -----------------------------
# Tokenized dataset (mask prompt tokens)
# -----------------------------

@dataclass
class TokenizationConfig:
    max_seq_length: int
    policy_id_for_training: str


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """
    Return start index of needle in haystack or -1.
    Used when chat template tokenization is not a strict prefix due to tokenizer quirks.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    # naive search is fine because sequences are short enough in practice
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


class SFTDataset(torch.utils.data.Dataset):
    """
    Pre-tokenized SFT dataset where labels mask the prompt portion (-100).
    """

    def __init__(
        self,
        examples: List[CleanTeacherExample],
        tokenizer: Any,
        cfg: DARTPipelineConfig,
        tok_cfg: TokenizationConfig,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tok_cfg = tok_cfg

        self.features: List[Dict[str, torch.Tensor]] = []
        self._build()

    def _build(self) -> None:
        max_len = int(self.tok_cfg.max_seq_length)
        policy_id = self.tok_cfg.policy_id_for_training

        iterator = self.examples
        if tqdm is not None:
            iterator = tqdm(self.examples, desc="tokenize", unit="ex")

        for ex in iterator:
            # Build prompt-only and full messages
            msgs_prompt = build_messages(self.cfg, policy_id, ex.prompt, assistant_target=None)
            msgs_full = build_messages(self.cfg, policy_id, ex.prompt, assistant_target=ex.target)

            prompt_text = render_chat(self.tokenizer, msgs_prompt, add_generation_prompt=True)
            full_text = render_chat(self.tokenizer, msgs_full, add_generation_prompt=False)

            # Tokenize without adding special tokens twice
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

            # Determine boundary where assistant target begins
            boundary = len(prompt_ids)
            if boundary > len(full_ids) or full_ids[:boundary] != prompt_ids:
                # Try to locate prompt_ids inside full_ids
                start = _find_subsequence(full_ids, prompt_ids)
                if start >= 0:
                    boundary = start + len(prompt_ids)
                else:
                    # Fallback: assume assistant begins near the end; mask everything except last segment
                    boundary = max(0, len(full_ids) - min(128, len(full_ids)))

            response_ids = full_ids[boundary:]
            if not response_ids:
                # Skip pathological cases
                continue

            # Truncate to max length by preserving response fully when possible
            if len(full_ids) > max_len:
                if len(response_ids) >= max_len:
                    # Keep only tail of response
                    response_ids = response_ids[-max_len:]
                    prompt_ids_tr = []
                else:
                    max_prompt = max_len - len(response_ids)
                    prompt_ids_tr = prompt_ids[-max_prompt:]
                input_ids = prompt_ids_tr + response_ids
                labels = [-100] * len(prompt_ids_tr) + response_ids
            else:
                input_ids = full_ids
                labels = [-100] * boundary + response_ids

            attention_mask = [1] * len(input_ids)

            self.features.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


class SFTDataCollator:
    """
    Pads input_ids/attention_mask/labels; labels use -100 pad.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]

        max_len = max(int(t.shape[0]) for t in input_ids)

        def pad(t: torch.Tensor, pad_id: int) -> torch.Tensor:
            if t.shape[0] == max_len:
                return t
            return torch.cat([t, torch.full((max_len - t.shape[0],), pad_id, dtype=t.dtype)], dim=0)

        input_ids_p = torch.stack([pad(t, self.tok.pad_token_id) for t in input_ids], dim=0)
        attention_p = torch.stack([pad(t, 0) for t in attention_mask], dim=0)
        labels_p = torch.stack([pad(t, -100) for t in labels], dim=0)

        return {"input_ids": input_ids_p, "attention_mask": attention_p, "labels": labels_p}


# -----------------------------
# Model loading & LoRA/QLoRA
# -----------------------------

def _dtype_from_cfg(cfg: DARTPipelineConfig) -> Optional[torch.dtype]:
    s = getattr(cfg.model, "torch_dtype", "auto")
    if not isinstance(s, str):
        return None
    s = s.lower().strip()
    if s in ("auto", ""):
        return None
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    return None


def _default_target_modules_for_llama() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoRAHyperparams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=_default_target_modules_for_llama)
    bias: str = "none"


@dataclass
class TrainHyperparams:
    # Core training
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"

    # Batch & perf
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Eval/save/log
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 20
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3


def train_hparams_from_cfg(cfg: DARTPipelineConfig) -> Tuple[TrainHyperparams, LoRAHyperparams]:
    """
    Best-effort extraction from config.py; falls back to safe defaults.
    """
    th = TrainHyperparams()
    lh = LoRAHyperparams()

    train_obj = getattr(cfg, "training", None)
    if train_obj is not None:
        for k in dataclasses.asdict(th).keys():
            if hasattr(train_obj, k):
                setattr(th, k, getattr(train_obj, k))
        # Optional: some configs name max_seq_length differently
        if hasattr(train_obj, "max_length"):
            th.max_seq_length = int(getattr(train_obj, "max_length"))

        # LoRA block
        if hasattr(train_obj, "lora_r"):
            lh.r = int(getattr(train_obj, "lora_r"))
        if hasattr(train_obj, "lora_alpha"):
            lh.alpha = int(getattr(train_obj, "lora_alpha"))
        if hasattr(train_obj, "lora_dropout"):
            lh.dropout = float(getattr(train_obj, "lora_dropout"))
        if hasattr(train_obj, "target_modules"):
            tm = getattr(train_obj, "target_modules")
            if isinstance(tm, (list, tuple)) and tm:
                lh.target_modules = list(tm)

    # Model block may also contain LoRA defaults
    model_obj = getattr(cfg, "model", None)
    if model_obj is not None:
        if hasattr(model_obj, "lora_r"):
            lh.r = int(getattr(model_obj, "lora_r"))
        if hasattr(model_obj, "lora_alpha"):
            lh.alpha = int(getattr(model_obj, "lora_alpha"))
        if hasattr(model_obj, "lora_dropout"):
            lh.dropout = float(getattr(model_obj, "lora_dropout"))
        if hasattr(model_obj, "lora_target_modules"):
            tm = getattr(model_obj, "lora_target_modules")
            if isinstance(tm, (list, tuple)) and tm:
                lh.target_modules = list(tm)

    # Precision sanity
    if not torch.cuda.is_available():
        th.bf16 = False
        th.fp16 = False

    return th, lh


def environment_diagnostics() -> Dict[str, Any]:
    """
    Collect lightweight environment info; logs nvidia-smi if available.

    IMPORTANT:
    On some clusters, LD_LIBRARY_PATH may point to an incompatible cuDNN runtime.
    Calling torch.backends.cudnn.version() triggers cuDNN initialization and can raise.
    We should not crash training just because diagnostics cannot be collected.
    """
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }

    # cuDNN diagnostics (best-effort; never crash)
    try:
        cudnn_avail = bool(torch.backends.cudnn.is_available())
        info["cudnn_available"] = cudnn_avail
        if cudnn_avail:
            info["cudnn_version"] = torch.backends.cudnn.version()
        else:
            info["cudnn_version"] = None
    except Exception as e:
        info["cudnn_available"] = None
        info["cudnn_version"] = None
        info["cudnn_error"] = repr(e)

    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info["nvidia_smi_stdout"] = r.stdout[:4000]
        info["nvidia_smi_stderr"] = r.stderr[:1000]
    except Exception as e:
        info["nvidia_smi_error"] = repr(e)

    return info


def load_tokenizer(base_model: str, tokenizer_name_or_path: Optional[str]) -> Any:
    tok_name = tokenizer_name_or_path or base_model
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True, trust_remote_code=getattr(tok_name, "trust_remote_code", False))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model_for_training(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    qlora: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> Any:
    """
    Load the base causal LM.
    - For standard LoRA: normal fp/bf16 load.
    - For QLoRA: bitsandbytes 4-bit or 8-bit.
    """
    if qlora:
        if BitsAndBytesConfig is None:
            raise RuntimeError("QLoRA requested but BitsAndBytesConfig is unavailable. Please install a compatible transformers/bitsandbytes stack.")
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requested but CUDA is not available.")

        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # default to 4bit for QLoRA
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            quantization_config=quant_cfg,
        )
        return model

    # Standard LoRA training (Trainer/Accelerate handles device placement)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    return model


def apply_or_load_lora(
    model: Any,
    lora_hp: LoRAHyperparams,
    *,
    init_adapter: Optional[str],
    qlora: bool,
    gradient_checkpointing: bool,
) -> Any:
    """
    Either:
    - Create fresh LoRA adapters, or
    - Load an existing adapter and continue SFT (DART-H).
    """
    # For k-bit training, prepare model (if helper exists)
    if qlora and prepare_model_for_kbit_training is not None:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        except Exception:
            pass

    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter)
        # Ensure adapter params are trainable
        try:
            model.train()
        except Exception:
            pass
        
        # ✅ 新增：显式将 LoRA 参数设为可训练
        for name, param in model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
        
        return model

    lora_cfg = LoraConfig(
        r=int(lora_hp.r),
        lora_alpha=int(lora_hp.alpha),
        lora_dropout=float(lora_hp.dropout),
        target_modules=list(lora_hp.target_modules),
        bias=str(lora_hp.bias),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model


def log_trainable_parameters(model: Any) -> Dict[str, Any]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = float(trainable) / float(total) if total > 0 else 0.0
    msg = f"Trainable parameters: {trainable:,} / {total:,} ({ratio*100:.2f}%)"
    logger.info(msg)
    return {"trainable": trainable, "total": total, "ratio": ratio}


# -----------------------------
# Training pipeline assembly
# -----------------------------

@dataclass
class TrainJob:
    stage: Literal["dart", "dart_h"]
    recipe: Literal["repair_only", "mix"]
    policy_id_for_training: str

    distill_path: Optional[Path]
    repair_path: Optional[Path]

    val_distill_path: Optional[Path]
    val_repair_path: Optional[Path]
    val_ratio: float

    # Oversampling
    repair_oversample: Dict[str, int]

    # Outputs
    output_dir: Path
    run_dir: Path

    # Model
    base_model: str
    init_adapter: Optional[str]
    qlora: bool
    load_in_4bit: bool
    load_in_8bit: bool


def build_run_dir(cfg: DARTPipelineConfig) -> Path:
    if isinstance(getattr(cfg, "derived", None), dict):
        p = cfg.derived.get("paths", {}).get("run_dir")
        if p:
            return Path(p)
    # Fallback: root/runs/exp_id
    root = Path(cfg.paths.root())
    exp_id = getattr(cfg.run, "exp_id", None) or "run"
    return root / "runs" / exp_id


def default_teacher_paths(run_dir: Path, stage: str) -> Tuple[Optional[Path], Optional[Path]]:
    distill = run_dir / "teacher_outputs" / "distill" / "outputs.jsonl"
    repair = run_dir / "teacher_outputs" / "repair" / "outputs.jsonl"
    if stage == "dart":
        return (distill if distill.exists() else None), None
    return (distill if distill.exists() else None), (repair if repair.exists() else None)


def dataset_statistics(examples: List[CleanTeacherExample]) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    by_task: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}

    for ex in examples:
        if ex.source:
            by_source[ex.source] = by_source.get(ex.source, 0) + 1
        if ex.gold_label:
            by_label[ex.gold_label] = by_label.get(ex.gold_label, 0) + 1
        if ex.task:
            by_task[ex.task] = by_task.get(ex.task, 0) + 1
        if ex.severity_bin:
            by_sev[ex.severity_bin] = by_sev.get(ex.severity_bin, 0) + 1

    return {
        "n": len(examples),
        "by_source": dict(sorted(by_source.items(), key=lambda x: (-x[1], x[0]))),
        "by_label": dict(sorted(by_label.items(), key=lambda x: (-x[1], x[0]))),
        "by_task": dict(sorted(by_task.items(), key=lambda x: (-x[1], x[0]))),
        "by_severity": dict(sorted(by_sev.items(), key=lambda x: (-x[1], x[0]))),
    }


def parse_oversample_map(s: Optional[str]) -> Dict[str, int]:
    """
    Example: "mild=1,moderate=2,severe=3,extreme=4"
    """
    if not s:
        return {}
    out: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        try:
            out[k] = max(1, int(v.strip()))
        except Exception:
            continue
    return out


def build_training_examples(job: TrainJob, cfg: DARTPipelineConfig, seed: int) -> Tuple[List[CleanTeacherExample], List[CleanTeacherExample]]:
    """
    Assemble (train, val) pairs according to stage/recipe.
    """
    distill_train: List[CleanTeacherExample] = []
    repair_train: List[CleanTeacherExample] = []
    distill_val: List[CleanTeacherExample] = []
    repair_val: List[CleanTeacherExample] = []

    if job.distill_path:
        distill_train = clean_teacher_outputs(job.distill_path)
    if job.repair_path:
        repair_train = clean_teacher_outputs(job.repair_path)

    if job.val_distill_path and job.val_distill_path.exists():
        distill_val = clean_teacher_outputs(job.val_distill_path)
    if job.val_repair_path and job.val_repair_path.exists():
        repair_val = clean_teacher_outputs(job.val_repair_path)

    distill_train = dedupe_examples(distill_train)
    repair_train = dedupe_examples(repair_train)
    distill_val = dedupe_examples(distill_val)
    repair_val = dedupe_examples(repair_val)

    # Apply oversampling to repair train set if configured
    if repair_train and job.repair_oversample:
        repair_train = oversample_by_severity(repair_train, job.repair_oversample)

    if job.stage == "dart":
        combined_train = distill_train
        combined_val = distill_val
    else:
        if job.recipe == "repair_only":
            combined_train = repair_train
            combined_val = repair_val
        else:
            # mix: D_orig ∪ λ D_repair (oversampling already applied)
            combined_train = distill_train + repair_train
            combined_val = distill_val + repair_val

    combined_train = dedupe_examples(combined_train)
    combined_val = dedupe_examples(combined_val)

    # If no explicit val set, split from training
    if not combined_val and job.val_ratio > 0.0:
        combined_train, combined_val = split_train_val(combined_train, job.val_ratio, seed=seed)

    return combined_train, combined_val


def build_training_arguments(cfg: DARTPipelineConfig, th: TrainHyperparams, output_dir: Path, run_name: str) -> TrainingArguments:
    """
    Create TrainingArguments with publication-grade defaults.

    Compatibility note:
    Different transformers versions use different keyword names (e.g., evaluation_strategy vs eval_strategy,
    older versions may have evaluate_during_training). We therefore build kwargs dynamically based on the
    actual TrainingArguments.__init__ signature to avoid TypeError.
    """
    _safe_mkdir(output_dir / "checkpoints")

    # Local import to avoid editing top-level imports
    import inspect

    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    params.discard("self")

    eval_strategy_val = str(th.eval_strategy) if th.eval_strategy else "no"
    do_eval_val = bool(th.eval_strategy and th.eval_strategy != "no")
    eval_steps_val = int(th.eval_steps) if do_eval_val else None

    # Base desired kwargs (we'll filter by signature below)
    want: Dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "overwrite_output_dir": True,

        "num_train_epochs": float(th.num_train_epochs),

        # Batch sizes (compat: per_device_* vs per_gpu_*)
        "per_device_train_batch_size": int(th.per_device_train_batch_size),
        "per_device_eval_batch_size": int(th.per_device_eval_batch_size),

        "gradient_accumulation_steps": int(th.gradient_accumulation_steps),

        "learning_rate": float(th.learning_rate),
        "weight_decay": float(th.weight_decay),
        "warmup_ratio": float(th.warmup_ratio),
        "lr_scheduler_type": str(th.lr_scheduler_type),
        "optim": str(th.optim),

        # Precision
        "bf16": bool(th.bf16),
        "fp16": bool(th.fp16),

        # Logging
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": int(th.logging_steps),
        "logging_first_step": True,
        "report_to": list(th.report_to),

        # Save strategy
        "save_strategy": str(th.save_strategy),
        "save_steps": int(th.save_steps),
        "save_total_limit": int(th.save_total_limit),

        # Best model selection (only meaningful if eval enabled)
        "load_best_model_at_end": bool(th.load_best_model_at_end) if do_eval_val else False,
        "metric_for_best_model": str(th.metric_for_best_model) if do_eval_val else None,
        "greater_is_better": bool(th.greater_is_better) if do_eval_val else None,

        # Performance knobs
        "gradient_checkpointing": bool(th.gradient_checkpointing),
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,

        # Reproducibility
        "seed": int(getattr(cfg.run, "seed", 42)),
        "data_seed": int(getattr(cfg.run, "seed", 42)),

        # Dataset provides labels; keep columns
        "remove_unused_columns": False,
        "label_names": ["labels"],

        "run_name": run_name,
    }

    # ---- Eval strategy compatibility ----
    # New/common: evaluation_strategy
    if "evaluation_strategy" in params:
        want["evaluation_strategy"] = eval_strategy_val
        want["eval_steps"] = eval_steps_val
    # Some versions: eval_strategy
    elif "eval_strategy" in params:
        want["eval_strategy"] = eval_strategy_val
        want["eval_steps"] = eval_steps_val
    # Older versions: evaluate_during_training + do_eval
    elif "evaluate_during_training" in params:
        want["evaluate_during_training"] = do_eval_val
        want["eval_steps"] = eval_steps_val

    # Also set do_eval if supported (helps older versions)
    if "do_eval" in params:
        want["do_eval"] = do_eval_val

    # ---- Batch size compatibility ----
    # If per_device_* is not supported, fall back to per_gpu_*
    if "per_device_train_batch_size" not in params and "per_gpu_train_batch_size" in params:
        want["per_gpu_train_batch_size"] = int(th.per_device_train_batch_size)
        want.pop("per_device_train_batch_size", None)

    if "per_device_eval_batch_size" not in params and "per_gpu_eval_batch_size" in params:
        want["per_gpu_eval_batch_size"] = int(th.per_device_eval_batch_size)
        want.pop("per_device_eval_batch_size", None)

    # Filter to only supported keys and drop None values (but keep False/0)
    filtered: Dict[str, Any] = {
        k: v for k, v in want.items()
        if (k in params) and (v is not None)
    }

    args = TrainingArguments(**filtered)
    return args


def save_final_artifacts(
    output_dir: Path,
    trainer: Trainer,
    tokenizer: Any,
    *,
    train_metrics: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """
    Save adapter/tokenizer plus metrics/summary into a stable structure.
    """
    final_dir = output_dir / "final"
    _safe_mkdir(final_dir)

    # Save adapter (PeftModel.save_pretrained saves only adapter weights/config)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Persist metrics and summary
    _json_dump_file(train_metrics, output_dir / "train.metrics.json")
    if eval_metrics is not None:
        _json_dump_file(eval_metrics, output_dir / "eval.metrics.json")
    _json_dump_file(summary, output_dir / "training_summary.json")


# -----------------------------
# CLI
# -----------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARTPipeline LoRA/QLoRA training (DART / DART-H).")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config. If omitted, defaults are used.")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotted overrides key=value.")
    p.add_argument("--materialize", action="store_true", help="Materialize runs/{exp_id}/ via config.materialize_run().")

    p.add_argument("--stage", type=str, required=True, choices=["dart", "dart_h"], help="Training stage.")
    p.add_argument("--recipe", type=str, default="repair_only", choices=["repair_only", "mix"],
                   help="For dart_h: repair_only or mix(distill + weighted repair).")

    p.add_argument("--adapter-name", type=str, required=True, help="Output adapter folder name under runs/{exp_id}/adapters/.")

    # Teacher inputs (optional overrides)
    p.add_argument("--distill-jsonl", type=str, default=None, help="Path to distill teacher outputs.jsonl.")
    p.add_argument("--repair-jsonl", type=str, default=None, help="Path to repair teacher outputs.jsonl.")
    p.add_argument("--val-distill-jsonl", type=str, default=None, help="Optional validation distill outputs.jsonl.")
    p.add_argument("--val-repair-jsonl", type=str, default=None, help="Optional validation repair outputs.jsonl.")
    p.add_argument("--val-ratio", type=float, default=0.05, help="If no val file, split this ratio from train deterministically.")

    # Oversampling
    p.add_argument("--repair-oversample", type=str, default="mild=1,moderate=2,severe=3,extreme=4",
                   help="Severity-based repeat map for repair examples, e.g. mild=1,moderate=2,severe=3,extreme=4. Use empty to disable.")

    # Policy choice for training prompt framing (default off; safe style is learned from targets)
    p.add_argument("--train-policy", type=str, default="off", choices=["on", "off"], help="System policy prompt used during training.")

    # Model
    p.add_argument("--base-model", type=str, default=None, help="Override base model path/name. Else use cfg.model.base_model_name_or_path.")
    p.add_argument("--init-adapter", type=str, default=None, help="Initialize from an existing adapter (for dart_h).")

    # QLoRA knobs
    p.add_argument("--qlora", action="store_true", help="Enable QLoRA (k-bit training). Requires bitsandbytes.")
    p.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization (QLoRA).")
    p.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization (QLoRA).")

    # Training hyperparams (override config defaults)
    p.add_argument("--epochs", type=float, default=None, help="num_train_epochs override.")
    p.add_argument("--lr", type=float, default=None, help="learning_rate override.")
    p.add_argument("--train-bsz", type=int, default=None, help="per_device_train_batch_size override.")
    p.add_argument("--eval-bsz", type=int, default=None, help="per_device_eval_batch_size override.")
    p.add_argument("--grad-accum", type=int, default=None, help="gradient_accumulation_steps override.")
    p.add_argument("--max-len", type=int, default=None, help="max_seq_length override.")
    p.add_argument("--eval-steps", type=int, default=None, help="eval_steps override.")
    p.add_argument("--save-steps", type=int, default=None, help="save_steps override.")
    p.add_argument("--logging-steps", type=int, default=None, help="logging_steps override.")
    p.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping even if config enables it.")

    # LoRA hyperparams overrides
    p.add_argument("--lora-r", type=int, default=None, help="LoRA rank r override.")
    p.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha override.")
    p.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout override.")
    p.add_argument("--lora-target-modules", type=str, default=None,
                   help="Comma-separated target modules (e.g., q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj).")

    return p.parse_args()


def main() -> None:
    args = _cli()

    # Load config
    if args.config:
        cfg = DARTPipelineConfig.from_dict(load_config_file(Path(args.config)))
    else:
        cfg = build_default_config()
    cfg = apply_overrides(cfg, args.override)

    # Materialize run directories if requested
    if args.materialize:
        cfg.materialize_run(create_dirs=True)
    else:
        # Ensure config consistency even if not materialized
        if hasattr(cfg, "data") and hasattr(cfg.data, "ensure_defaults"):
            cfg.data.ensure_defaults(cfg.paths)
        if hasattr(cfg, "validate"):
            cfg.validate()

    run_dir = build_run_dir(cfg)
    adapters_dir = run_dir / "adapters"
    _safe_mkdir(adapters_dir)

    # Logging
    logs_dir = run_dir / "logs"
    configure_logging(logs_dir, level=logging.INFO, console=True)

    # Seed
    seed = int(getattr(cfg.run, "seed", 42))
    set_global_seed(seed, deterministic=bool(getattr(cfg.run, "deterministic", True)))

    # Diagnostics snapshot
    env_info = environment_diagnostics()
    logger.info("Environment diagnostics: cuda_available=%s cuda_version=%s", env_info.get("cuda_available"), env_info.get("cuda_version"))

    # Defaults for teacher paths
    default_distill, default_repair = default_teacher_paths(run_dir, args.stage)

    distill_path = Path(args.distill_jsonl).expanduser().resolve() if args.distill_jsonl else default_distill
    repair_path = Path(args.repair_jsonl).expanduser().resolve() if args.repair_jsonl else default_repair

    val_distill_path = Path(args.val_distill_jsonl).expanduser().resolve() if args.val_distill_jsonl else None
    val_repair_path = Path(args.val_repair_jsonl).expanduser().resolve() if args.val_repair_jsonl else None

    # Training policy id
    policy_id_for_training = cfg.policy.policy_id_on if args.train_policy == "on" else cfg.policy.policy_id_off

    # Output dir
    output_dir = adapters_dir / args.adapter_name
    _safe_mkdir(output_dir)

    # Model base
    base_model = args.base_model or getattr(cfg.model, "base_model_name_or_path", None)
    if not base_model:
        raise RuntimeError("Base model is not specified. Provide --base-model or set cfg.model.base_model_name_or_path.")

    # Stage sanity
    if args.stage == "dart" and not distill_path:
        raise FileNotFoundError("DART stage requires distill teacher outputs. Provide --distill-jsonl or generate teacher_outputs/distill/outputs.jsonl.")
    if args.stage == "dart_h":
        if not repair_path:
            raise FileNotFoundError("DART-H stage requires repair teacher outputs. Provide --repair-jsonl or generate teacher_outputs/repair/outputs.jsonl.")
        if not args.init_adapter:
            logger.warning("DART-H typically continues from DART adapter; you did not specify --init-adapter. Training will start from a fresh LoRA init unless your config sets otherwise.")

    # Parse oversample map
    repair_oversample = parse_oversample_map(args.repair_oversample)

    # Resolve init-adapter placeholders and validate path exists (prevents PEFT treating it as Hub repo id)
    init_adapter = args.init_adapter
    if init_adapter:
        init_adapter = os.path.expandvars(init_adapter)

        exp_id = getattr(cfg.run, "exp_id", None) or os.environ.get("EXP_ID")
        if exp_id:
            init_adapter = (
                init_adapter
                .replace("{exp_id}", exp_id)
                .replace("{EXP_ID}", exp_id)
                .replace("<exp_id>", exp_id)
            )

        p = Path(init_adapter)
        if not p.is_absolute():
            p = (Path.cwd() / p)

        if not (p.exists() and p.is_dir()):
            raise FileNotFoundError(
                f"--init-adapter path does not exist: {str(p)}. "
                f"Did you forget to substitute exp_id? Example: runs/{exp_id}/adapters/DART/final"
            )

        init_adapter = str(p.resolve())

    job = TrainJob(
        stage=args.stage,
        recipe=args.recipe,
        policy_id_for_training=policy_id_for_training,
        distill_path=distill_path,
        repair_path=repair_path,
        val_distill_path=val_distill_path,
        val_repair_path=val_repair_path,
        val_ratio=float(args.val_ratio),
        repair_oversample=repair_oversample,
        output_dir=output_dir,
        run_dir=run_dir,
        base_model=base_model,
        init_adapter=init_adapter,
        qlora=bool(args.qlora),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
    )

    # Load hyperparams from cfg and apply CLI overrides
    th, lh = train_hparams_from_cfg(cfg)

    if args.epochs is not None:
        th.num_train_epochs = float(args.epochs)
    if args.lr is not None:
        th.learning_rate = float(args.lr)
    if args.train_bsz is not None:
        th.per_device_train_batch_size = int(args.train_bsz)
    if args.eval_bsz is not None:
        th.per_device_eval_batch_size = int(args.eval_bsz)
    if args.grad_accum is not None:
        th.gradient_accumulation_steps = int(args.grad_accum)
    if args.max_len is not None:
        th.max_seq_length = int(args.max_len)
    if args.eval_steps is not None:
        th.eval_steps = int(args.eval_steps)
    if args.save_steps is not None:
        th.save_steps = int(args.save_steps)
    if args.logging_steps is not None:
        th.logging_steps = int(args.logging_steps)

    if args.lora_r is not None:
        lh.r = int(args.lora_r)
    if args.lora_alpha is not None:
        lh.alpha = int(args.lora_alpha)
    if args.lora_dropout is not None:
        lh.dropout = float(args.lora_dropout)
    if args.lora_target_modules is not None:
        tm = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        if tm:
            lh.target_modules = tm

    if args.no_early_stopping:
        th.early_stopping = False

    # Build training data
    train_examples, val_examples = build_training_examples(job, cfg, seed=seed)

    if not train_examples:
        raise RuntimeError("No training examples after filtering. Check teacher outputs, success flags, and conclusion formatting.")

    logger.info("Train examples: %d | Val examples: %d", len(train_examples), len(val_examples))
    logger.info("Train stats: %s", json.dumps(dataset_statistics(train_examples), ensure_ascii=False))
    if val_examples:
        logger.info("Val stats: %s", json.dumps(dataset_statistics(val_examples), ensure_ascii=False))

    # Save manifests / snapshot
    manifest = {
        "timestamp_utc": utc_now_iso(),
        "stage": job.stage,
        "recipe": job.recipe,
        "policy_id_for_training": job.policy_id_for_training,
        "inputs": {
            "distill": file_manifest(job.distill_path) if job.distill_path else None,
            "repair": file_manifest(job.repair_path) if job.repair_path else None,
            "val_distill": file_manifest(job.val_distill_path) if job.val_distill_path else None,
            "val_repair": file_manifest(job.val_repair_path) if job.val_repair_path else None,
        },
        "repair_oversample": job.repair_oversample,
        "env": env_info,
    }
    _json_dump_file(manifest, output_dir / "data.manifest.json")

    snapshot = {
        "timestamp_utc": utc_now_iso(),
        "job": dataclasses.asdict(job),
        "train_hparams": dataclasses.asdict(th),
        "lora_hparams": dataclasses.asdict(lh),
        "config_run": dataclasses.asdict(cfg.run),
        "config_model": dataclasses.asdict(cfg.model),
        "config_policy": dataclasses.asdict(cfg.policy),
    }
    _json_dump_file(snapshot, output_dir / "job.snapshot.json")

    # Load tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(cfg.model, "tokenizer_name_or_path", None) or base_model,
        use_fast=True,
        trust_remote_code=bool(getattr(cfg.model, "trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_cfg = TokenizationConfig(
        max_seq_length=int(th.max_seq_length),
        policy_id_for_training=job.policy_id_for_training,
    )
    train_ds = SFTDataset(train_examples, tokenizer, cfg, tok_cfg)
    eval_ds = SFTDataset(val_examples, tokenizer, cfg, tok_cfg) if val_examples else None

    logger.info("Tokenized train size: %d | tokenized val size: %s", len(train_ds), (len(eval_ds) if eval_ds else "None"))

    # Load base model and apply/continue LoRA
    torch_dtype = _dtype_from_cfg(cfg)
    trust_remote_code = bool(getattr(cfg.model, "trust_remote_code", False))

    model = load_base_model_for_training(
        base_model_name_or_path=base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        qlora=job.qlora,
        load_in_4bit=job.load_in_4bit,
        load_in_8bit=job.load_in_8bit,
    )

    model = apply_or_load_lora(
        model,
        lora_hp=lh,
        init_adapter=job.init_adapter,
        qlora=job.qlora,
        gradient_checkpointing=bool(th.gradient_checkpointing),
    )

    trainable_info = log_trainable_parameters(model)

    # Training arguments
    run_name = f"{job.stage}_{args.adapter_name}"
    training_args = build_training_arguments(cfg, th, output_dir, run_name=run_name)

    # Collator
    collator = SFTDataCollator(tokenizer)

    # Callbacks
    callbacks = []
    if th.early_stopping and eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(th.early_stopping_patience)))

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training: stage=%s recipe=%s qlora=%s", job.stage, job.recipe, job.qlora)
    t0 = time.time()
    train_result = trainer.train()
    t1 = time.time()

    train_metrics = dict(train_result.metrics or {})
    train_metrics.update(
        {
            "train_runtime_sec": t1 - t0,
            "train_samples": len(train_ds),

            # Flatten trainable_info (Trainer.log_metrics requires scalar-like values)
            "trainable_params": int(trainable_info.get("trainable", 0)),
            "total_params": int(trainable_info.get("total", 0)),
            "trainable_ratio": float(trainable_info.get("ratio", 0.0)),
        }
    )

    # Defensive: only log scalar-like values to avoid formatting crash
    train_metrics_for_log = {
        k: v for k, v in train_metrics.items()
        if isinstance(v, (int, float, str, bool))
    }

    trainer.log_metrics("train", train_metrics_for_log)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = None
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # Save final adapter/tokenizer + summary
    summary = {
        "timestamp_utc": utc_now_iso(),
        "stage": job.stage,
        "recipe": job.recipe,
        "adapter_name": args.adapter_name,
        "base_model": base_model,
        "init_adapter": job.init_adapter,
        "policy_id_for_training": job.policy_id_for_training,
        "qlora": job.qlora,
        "train_size": len(train_ds),
        "val_size": (len(eval_ds) if eval_ds is not None else 0),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_hparams": dataclasses.asdict(th),
        "lora_hparams": dataclasses.asdict(lh),
        "seed": seed,
    }

    save_final_artifacts(
        output_dir=output_dir,
        trainer=trainer,
        tokenizer=tokenizer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        summary=summary,
    )

    logger.info("Training complete. Adapter saved to: %s", str(output_dir / "final"))


if __name__ == "__main__":
    # Needed for regex in teacher output parsing
    import re

    main()