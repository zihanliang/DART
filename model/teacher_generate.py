# model/teacher_generate.py
"""
Teacher generation for DARTPipeline (Audit → Repair → Policy).

This module generates:
1) Distillation targets for DART training:
   - Input: normalized JSONL samples with gold_label in {YES, NO}
   - Output: short analysis (2–4 sentences) + "Conclusion: YES/NO" aligned with gold label

2) Repair safe targets for DART-H:
   - Input: regression pool JSONL (from audit.py) or any JSONL containing prompt + gold_label
   - Output: "Safe Analysis Targets" that avoid repeating/elaborating hateful/violent/toxic content
            while preserving correct conclusion label

Engineering features (ACL-level reproducibility & robustness):
- Async OpenAI-compatible chat-completions client (DeepSeek style supported)
- Dual-rate limiting (RPM + TPM) using token bucket
- Exponential backoff retries, error typing
- Strict output parsing + format correction attempts
- Deterministic processing order (sorted by sample_id)
- Checkpoint/resume (processed_ids + errors)
- Output JSONL with full provenance and per-sample metadata
- Job snapshot and config provenance under runs/{exp_id}/teacher_outputs/{task}/

Assumed repo layout:
  project_root/
    model/config.py
    runs/{exp_id}/datasets/... (from model/datasets.py export)
    runs/{exp_id}/repair_pool/... (from model/audit.py export)
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
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

# Optional progress bar
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# HTTP client
try:
    import aiohttp  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("aiohttp is required for teacher_generate.py. Please install aiohttp.") from e

# Import config with both package and script compatibility.
try:
    from .config import (
        DARTPipelineConfig,
        TeacherConfig,
        GenerationConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        _json_dump_file,  # type: ignore
        _safe_mkdir,  # type: ignore
    )
except Exception:  # pragma: no cover
    from config import (  # type: ignore
        DARTPipelineConfig,
        TeacherConfig,
        GenerationConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        _json_dump_file,
        _safe_mkdir,
    )

logger = logging.getLogger("dartpipeline.teacher_generate")


# -----------------------------
# Data IO (normalized JSONL)
# -----------------------------

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -----------------------------
# Label normalization
# -----------------------------

GoldLabel = Literal["YES", "NO"]


def normalize_gold_label(rec: Dict[str, Any]) -> Optional[GoldLabel]:
    """
    Derive gold_label from record fields and meta where possible.
    """
    gl = rec.get("gold_label")
    if isinstance(gl, str):
        gl_u = gl.strip().upper()
        if gl_u in ("YES", "NO"):
            return gl_u  # type: ignore[return-value]
        if gl_u in ("DIFF", "EQUAL"):
            return "YES" if gl_u == "DIFF" else "NO"

    meta = rec.get("meta") or {}
    if isinstance(meta, dict):
        # common meta hints from our normalization
        if "requires_difference_awareness" in meta:
            return "YES" if bool(meta["requires_difference_awareness"]) else "NO"
        if "label_int" in meta:
            try:
                return "YES" if int(meta["label_int"]) == 1 else "NO"
            except Exception:
                pass
        if "condition" in meta and isinstance(meta["condition"], str):
            c = meta["condition"].lower().strip()
            if c == "diff":
                return "YES"
            if c == "equal":
                return "NO"

    return None


def record_sample_id(rec: Dict[str, Any]) -> str:
    sid = rec.get("sample_id")
    if sid is None:
        # fallback deterministic-ish
        prompt = str(rec.get("prompt") or "")
        return f"missing_id:{hash(prompt)}"
    return str(sid)


def record_prompt(rec: Dict[str, Any]) -> str:
    p = rec.get("prompt")
    if p is None:
        # fallback to other common fields
        for k in ["question", "text", "input"]:
            if k in rec:
                return str(rec[k])
        return ""
    return str(p)


# -----------------------------
# Output parsing (strict)
# -----------------------------

@dataclass
class ParsedTeacherOutput:
    raw_text: str
    analysis: str
    conclusion: Optional[GoldLabel]
    parse_ok: bool
    issues: List[str] = field(default_factory=list)


def parse_teacher_output(text: str, gen_cfg: GenerationConfig) -> ParsedTeacherOutput:
    """
    Parse teacher output enforcing:
      - contains "Conclusion: YES|NO"
      - analysis is everything before the conclusion line (best-effort)
    """
    raw = (text or "").strip()
    issues: List[str] = []
    if not raw:
        return ParsedTeacherOutput(raw_text="", analysis="", conclusion=None, parse_ok=False, issues=["empty_output"])

    m = re.search(gen_cfg.conclusion_regex, raw, flags=re.IGNORECASE)
    if not m:
        return ParsedTeacherOutput(raw_text=raw, analysis=raw, conclusion=None, parse_ok=False, issues=["missing_conclusion_line"])

    concl = m.group(1).upper()
    if concl not in ("YES", "NO"):
        issues.append("invalid_conclusion_value")
        conclusion = None
    else:
        conclusion = concl  # type: ignore[assignment]

    # Extract analysis
    analysis = raw
    ms = re.match(gen_cfg.analysis_split_regex, raw, flags=re.IGNORECASE | re.DOTALL)
    if ms:
        analysis = (ms.group(1) or "").strip()
    else:
        # Fallback: take everything before the matched conclusion start index
        analysis = raw[: m.start()].strip()

    if not analysis:
        issues.append("empty_analysis")

    parse_ok = (conclusion is not None) and ("missing_conclusion_line" not in issues)
    return ParsedTeacherOutput(raw_text=raw, analysis=analysis, conclusion=conclusion, parse_ok=parse_ok, issues=issues)


def sentence_count(text: str) -> int:
    """
    Conservative sentence count for English-like punctuation.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p for p in parts if p.strip()]
    return len(parts)


def contains_long_quote_of_prompt(prompt: str, analysis: str, min_chars: int = 40) -> bool:
    """
    Heuristic: detect long verbatim quotes from prompt.
    We avoid copying potentially toxic content into the explanation.

    This is a conservative check: if analysis contains any substring from prompt of length >= min_chars.
    For efficiency, we sample prompt windows.
    """
    p = (prompt or "").strip()
    a = (analysis or "").strip()
    if len(p) < min_chars or len(a) < min_chars:
        return False

    # Normalize whitespace
    p_norm = re.sub(r"\s+", " ", p)
    a_norm = re.sub(r"\s+", " ", a)

    # Sample windows from prompt
    step = max(10, min_chars // 2)
    for i in range(0, max(0, len(p_norm) - min_chars + 1), step):
        chunk = p_norm[i : i + min_chars]
        if chunk and chunk in a_norm:
            return True
    return False


# -----------------------------
# Rate limiting & retries
# -----------------------------

class RetryableError(Exception):
    pass


class NonRetryableError(Exception):
    pass


class TokenBucket:
    """
    Async token bucket rate limiter.
    """

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_ts = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_ts
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_ts = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                wait_s = (tokens - self.tokens) / self.rate if self.rate > 0 else 1.0
                await asyncio.sleep(max(0.01, wait_s))


class DualLimiter:
    """
    Dual limiter for requests-per-minute and tokens-per-minute.
    """

    def __init__(self, rpm: int, tpm: int) -> None:
        # RPM -> per second
        rps = max(1e-6, rpm / 60.0)
        tps = max(1e-6, tpm / 60.0)
        self.req_bucket = TokenBucket(rate_per_sec=rps, capacity=rps * 2.0)
        self.tok_bucket = TokenBucket(rate_per_sec=tps, capacity=tps * 2.0)

    async def acquire(self, est_tokens: int) -> None:
        await self.req_bucket.acquire(1.0)
        await self.tok_bucket.acquire(float(max(1, est_tokens)))


async def with_retries(
    coro_fn,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    retry_on: Tuple[type, ...] = (RetryableError,),
) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except retry_on as e:
            last_err = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2.0 ** attempt))
            logger.warning("Retryable error: %s | attempt=%d/%d | sleep=%.2fs", str(e), attempt + 1, max_retries + 1, delay)
            await asyncio.sleep(delay)
        except NonRetryableError:
            raise
        except Exception as e:
            # Treat unknown errors as retryable up to max_retries
            last_err = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2.0 ** attempt))
            logger.warning("Unknown error (treated retryable): %s | attempt=%d/%d | sleep=%.2fs", str(e), attempt + 1, max_retries + 1, delay)
            await asyncio.sleep(delay)
    raise last_err if last_err is not None else RuntimeError("Unknown failure")


# -----------------------------
# OpenAI-compatible async client
# -----------------------------

@dataclass
class ChatClientConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int
    max_tokens: int
    temperature: float
    max_retries: int
    backoff_base_seconds: float
    backoff_max_seconds: float
    rpm: int
    tpm: int


class OpenAICompatibleChatClient:
    """
    Minimal OpenAI-compatible / DeepSeek-compatible client using /chat/completions endpoint.
    """

    def __init__(self, cfg: ChatClientConfig) -> None:
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        self.limiter = DualLimiter(rpm=cfg.rpm, tpm=cfg.tpm)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_seconds)
            connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _estimate_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> int:
        # Very rough heuristic: 1 token ~ 4 chars English-ish
        msg_chars = sum(len(m.get("content", "")) for m in messages)
        return int(msg_chars / 4) + int(max_tokens)

    async def chat(self, messages: List[Dict[str, str]], *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        mt = int(max_tokens if max_tokens is not None else self.cfg.max_tokens)
        temp = float(temperature if temperature is not None else self.cfg.temperature)

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": mt,
            "temperature": temp,
        }

        est = self._estimate_tokens(messages, mt)
        await self.limiter.acquire(est_tokens=est)

        session = await self._get_session()

        async def _do() -> Dict[str, Any]:
            try:
                async with session.post(self.url, json=payload, headers=self.headers) as resp:
                    txt = await resp.text()
                    if resp.status == 200:
                        data = json.loads(txt)
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        model = data.get("model", self.cfg.model)
                        return {"success": True, "content": content, "usage": usage, "model": model}
                    if resp.status == 429 or resp.status >= 500:
                        raise RetryableError(f"HTTP {resp.status}: {txt[:300]}")
                    raise NonRetryableError(f"HTTP {resp.status}: {txt[:500]}")
            except asyncio.TimeoutError:
                raise RetryableError("timeout")
            except aiohttp.ClientError as e:
                raise RetryableError(f"client_error: {e}")

        return await with_retries(
            _do,
            max_retries=self.cfg.max_retries,
            base_delay=self.cfg.backoff_base_seconds,
            max_delay=self.cfg.backoff_max_seconds,
            retry_on=(RetryableError,),
        )


# -----------------------------
# Checkpointing
# -----------------------------

@dataclass
class TeacherCheckpoint:
    task: str
    total: int
    processed_count: int = 0
    processed_ids: Dict[str, bool] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    last_update_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TeacherCheckpoint":
        return TeacherCheckpoint(
            task=str(d.get("task", "")),
            total=int(d.get("total", 0)),
            processed_count=int(d.get("processed_count", 0)),
            processed_ids=dict(d.get("processed_ids", {})),
            errors=list(d.get("errors", [])),
            last_update_utc=str(d.get("last_update_utc", "")),
        )


class CheckpointManager:
    def __init__(self, ckpt_path: Path) -> None:
        self.ckpt_path = ckpt_path
        _safe_mkdir(ckpt_path.parent)

    def load(self) -> Optional[TeacherCheckpoint]:
        if not self.ckpt_path.exists():
            return None
        try:
            obj = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
            return TeacherCheckpoint.from_dict(obj)
        except Exception:
            return None

    def save(self, ckpt: TeacherCheckpoint) -> None:
        ckpt.last_update_utc = utc_now_iso()
        tmp = self.ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.ckpt_path)


# -----------------------------
# Prompt templates (distill & repair)
# -----------------------------

def build_distill_system_prompt(cfg: DARTPipelineConfig) -> str:
    # We reuse policy-style constraints but do NOT require the model to be "policy on"—this is teacher target quality control.
    return (
        "You are writing a brief explanation for a labeled difference-awareness classification example.\n"
        "Write a concise analysis (2–4 sentences). Avoid unnecessary details.\n"
        "Finish with a single line exactly in the form: Conclusion: YES or Conclusion: NO.\n"
    )


def build_repair_system_prompt(cfg: DARTPipelineConfig) -> str:
    # Stronger constraints aligned with DARTPipeline repair stage (safe analysis targets).
    return (
        "You are writing a SAFE brief explanation for a labeled difference-awareness classification example.\n"
        "Write a concise analysis (2–4 sentences).\n"
        "Do NOT repeat, quote, or elaborate on hateful/violent/toxic content that may appear in the prompt.\n"
        "Avoid slurs and graphic details. Use high-level, abstract wording.\n"
        "Finish with a single line exactly in the form: Conclusion: YES or Conclusion: NO.\n"
    )


def build_user_prompt(
    task: Literal["distill", "repair"],
    prompt: str,
    gold_label: GoldLabel,
) -> str:
    """
    Teacher-forced labeling: we provide the gold label to ensure correctness, but instruct the teacher
    to write as if it is the decision outcome.
    """
    label_sentence = "YES" if gold_label == "YES" else "NO"
    if task == "distill":
        return (
            "Task: Determine whether this prompt requires difference-awareness.\n"
            f"Gold label: {label_sentence}.\n"
            "Write a short analysis consistent with the gold label, then output the conclusion line.\n\n"
            f"Prompt:\n{prompt}"
        )
    # repair
    return (
        "Task: Produce a SAFE short analysis consistent with the gold label.\n"
        "Important: Do not repeat or expand any potentially toxic/hateful content from the prompt.\n"
        f"Gold label: {label_sentence}.\n"
        "Write the analysis, then output the conclusion line.\n\n"
        f"Prompt:\n{prompt}"
    )


def build_format_fix_user_prompt(expected_label: GoldLabel) -> str:
    """
    Correction instruction used when teacher output violates format or label.
    """
    return (
        "Your previous response did not follow the required format.\n"
        "Requirements:\n"
        "- Analysis must be 2–4 sentences.\n"
        "- End with exactly one line: Conclusion: YES or Conclusion: NO.\n"
        f"- The conclusion MUST be: Conclusion: {expected_label}.\n"
        "Please output the corrected final answer only."
    )


# -----------------------------
# Core generation logic
# -----------------------------

@dataclass
class TeacherJobConfig:
    task: Literal["distill", "repair"]
    input_path: Path
    output_path: Path
    checkpoint_path: Path
    job_snapshot_path: Path

    # Processing filters
    splits: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    max_samples: Optional[int] = None

    # Concurrency / resume
    concurrency: int = 16
    save_every: int = 50
    resume: bool = True
    reprocess_failed: bool = False

    # Output quality control
    format_fix_rounds: int = 2
    enforce_sentence_range: bool = True
    min_sentences: int = 2
    max_sentences: int = 4
    avoid_prompt_quoting: bool = True
    quote_min_chars: int = 40


def filter_records(records: List[Dict[str, Any]], job: TeacherJobConfig) -> List[Dict[str, Any]]:
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
    # Deterministic order
    out.sort(key=lambda x: record_sample_id(x))
    if job.max_samples is not None and len(out) > job.max_samples:
        out = out[: job.max_samples]
    return out


def load_existing_processed(output_path: Path) -> Dict[str, bool]:
    """
    Read output JSONL to build sample_id -> success map.
    """
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
                success = bool(obj.get("success", False))
                processed[sid] = success
            except Exception:
                continue
    return processed


async def generate_one(
    client: OpenAICompatibleChatClient,
    cfg: DARTPipelineConfig,
    gen_cfg: GenerationConfig,
    job: TeacherJobConfig,
    rec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate teacher target for one record with format enforcement and optional corrections.

    Pipeline alignment:
      - Preserve provenance (source/split/meta) from input records.
      - For repair jobs, audit exports `severity_bin` at the top level; we copy it into meta so
        train_lora can oversample by severity in Step 7.2.
    """
    sid = record_sample_id(rec)
    prompt = record_prompt(rec)
    gold = normalize_gold_label(rec)

    # Propagate provenance/meta from the input record
    meta_in = rec.get("meta")
    meta: Dict[str, Any] = dict(meta_in) if isinstance(meta_in, dict) else {}

    # For repair inputs exported by audit.py, severity_bin is top-level. Copy into meta for oversampling.
    if rec.get("severity_bin") is not None and "severity_bin" not in meta:
        meta["severity_bin"] = rec.get("severity_bin")
    if rec.get("regression_trigger") is not None and "regression_trigger" not in meta:
        meta["regression_trigger"] = rec.get("regression_trigger")

    base_meta: Dict[str, Any] = {
        "source": rec.get("source"),
        "split": rec.get("split"),
        "meta": meta,
    }
    # Keep these top-level too (useful for debugging / analysis; harmless for distill jobs).
    if rec.get("severity_bin") is not None:
        base_meta["severity_bin"] = rec.get("severity_bin")

    if not prompt:
        return {
            "sample_id": sid,
            "task": job.task,
            "success": False,
            "error": "empty_prompt",
            "timestamp_utc": utc_now_iso(),
            **base_meta,
        }
    if gold is None:
        return {
            "sample_id": sid,
            "task": job.task,
            "success": False,
            "error": "missing_gold_label",
            "timestamp_utc": utc_now_iso(),
            **base_meta,
        }

    system_prompt = build_distill_system_prompt(cfg) if job.task == "distill" else build_repair_system_prompt(cfg)
    user_prompt = build_user_prompt(job.task, prompt, gold)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # First attempt (do not pass placeholder args; max_tokens is managed by client cfg)
    resp = await client.chat(messages)

    raw_text = resp.get("content", "") if isinstance(resp, dict) else ""
    parsed = parse_teacher_output(raw_text, gen_cfg)

    def _violations(p: ParsedTeacherOutput) -> List[str]:
        v: List[str] = []
        if not p.parse_ok:
            v.extend(p.issues)
        if p.conclusion is None:
            v.append("missing_conclusion")
        elif p.conclusion != gold:
            v.append("wrong_conclusion")
        if job.enforce_sentence_range:
            sc = sentence_count(p.analysis)
            if sc < job.min_sentences or sc > job.max_sentences:
                v.append(f"sentence_count_out_of_range:{sc}")
        if job.avoid_prompt_quoting and contains_long_quote_of_prompt(prompt, p.analysis, min_chars=job.quote_min_chars):
            v.append("analysis_quotes_prompt")
        return v

    violations = _violations(parsed)

    # Correction rounds
    rounds_used = 0
    while violations and rounds_used < job.format_fix_rounds:
        rounds_used += 1
        fix_user = build_format_fix_user_prompt(gold)
        messages_fix = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": parsed.raw_text},
            {"role": "user", "content": fix_user},
        ]
        resp = await client.chat(messages_fix)
        raw_text = resp.get("content", "") if isinstance(resp, dict) else ""
        parsed = parse_teacher_output(raw_text, gen_cfg)
        violations = _violations(parsed)

    success = (not violations) and parsed.parse_ok and (parsed.conclusion == gold)

    out = {
        "sample_id": sid,
        "task": job.task,
        "teacher_model": client.cfg.model,
        "teacher_provider": cfg.teacher.provider,
        "timestamp_utc": utc_now_iso(),
        "success": success,
        "gold_label": gold,
        "prompt": prompt,
        "messages": messages,  # first-turn messages for provenance
        "raw_text": parsed.raw_text,
        "parsed_analysis": parsed.analysis,
        "parsed_conclusion": parsed.conclusion,
        "parse_ok": parsed.parse_ok,
        "parse_issues": parsed.issues,
        "violations": violations,
        "format_fix_rounds_used": rounds_used,
        "usage": resp.get("usage", {}) if isinstance(resp, dict) else {},
        **base_meta,
    }
    return out


# -----------------------------
# Building client from DARTPipelineConfig
# -----------------------------

def build_client_from_config(cfg: DARTPipelineConfig) -> OpenAICompatibleChatClient:
    """
    Create an OpenAI-compatible client using TeacherConfig fields and environment variables.
    """
    api_key = os.getenv(cfg.teacher.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {cfg.teacher.api_key_env}")

    base_url = os.getenv(cfg.teacher.base_url_env, "").strip()
    if not base_url:
        # Provide sensible defaults by provider
        if cfg.teacher.provider.lower() == "deepseek":
            base_url = "https://api.deepseek.com/v1"
        else:
            raise RuntimeError(
                f"Missing base URL environment variable: {cfg.teacher.base_url_env}. "
                "Set it to an OpenAI-compatible endpoint base URL (e.g., https://.../v1)."
            )

    # Map teacher config to client config
    ccfg = ChatClientConfig(
        api_key=api_key,
        base_url=base_url,
        model=cfg.teacher.model,
        timeout_seconds=cfg.teacher.timeout_seconds,
        max_tokens=512,  # teacher max tokens for short rationales
        temperature=0.2,  # low variance for reproducibility
        max_retries=cfg.teacher.max_retries,
        backoff_base_seconds=cfg.teacher.backoff_base_seconds,
        backoff_max_seconds=cfg.teacher.backoff_max_seconds,
        rpm=cfg.teacher.requests_per_minute,
        tpm=100000,  # tokens-per-minute; keep high but controlled by provider; can be overridden later if needed
    )
    return OpenAICompatibleChatClient(ccfg)


# -----------------------------
# Main runner
# -----------------------------

async def run_teacher_job(cfg: DARTPipelineConfig, job: TeacherJobConfig) -> None:
    """
    Run a teacher generation job end-to-end with checkpointing and concurrency.
    """
    _safe_mkdir(job.output_path.parent)
    _safe_mkdir(job.checkpoint_path.parent)

    # Load records
    if not job.input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {job.input_path}")
    records = read_jsonl(job.input_path)
    records = filter_records(records, job)

    if not records:
        logger.warning("No records after filtering. Nothing to do.")
        return

    # Resume logic: combine checkpoint + existing output file
    processed_from_output = load_existing_processed(job.output_path)
    ckpt_mgr = CheckpointManager(job.checkpoint_path)
    ckpt = ckpt_mgr.load() if job.resume else None
    if ckpt is None:
        ckpt = TeacherCheckpoint(task=job.task, total=len(records))

    # Merge processed ids
    for sid, ok in processed_from_output.items():
        if sid not in ckpt.processed_ids:
            ckpt.processed_ids[sid] = ok

    # Build todo list
    todo: List[Dict[str, Any]] = []
    for r in records:
        sid = record_sample_id(r)
        if sid in ckpt.processed_ids:
            if job.reprocess_failed and ckpt.processed_ids.get(sid) is False:
                todo.append(r)
            else:
                continue
        else:
            todo.append(r)

    logger.info("Teacher job '%s': total=%d | todo=%d | already=%d",
                job.task, len(records), len(todo), len(records) - len(todo))

    # Write job snapshot
    snapshot = {
        "task": job.task,
        "input_path": str(job.input_path),
        "output_path": str(job.output_path),
        "checkpoint_path": str(job.checkpoint_path),
        "filters": {"splits": job.splits, "sources": job.sources, "max_samples": job.max_samples},
        "concurrency": job.concurrency,
        "resume": job.resume,
        "reprocess_failed": job.reprocess_failed,
        "quality_control": {
            "format_fix_rounds": job.format_fix_rounds,
            "enforce_sentence_range": job.enforce_sentence_range,
            "min_sentences": job.min_sentences,
            "max_sentences": job.max_sentences,
            "avoid_prompt_quoting": job.avoid_prompt_quoting,
            "quote_min_chars": job.quote_min_chars,
        },
        "teacher_config": dataclasses.asdict(cfg.teacher),
        "generation_config": dataclasses.asdict(cfg.generation),
        "run": dataclasses.asdict(cfg.run),
        "timestamp_utc": utc_now_iso(),
    }
    _json_dump_file(snapshot, job.job_snapshot_path)

    # Create client
    client = build_client_from_config(cfg)
    gen_cfg = cfg.generation

    sem = asyncio.Semaphore(job.concurrency)

    # Progress bar
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(todo), desc=f"teacher:{job.task}", unit="sample")

    processed_since_save = 0

    async def _worker_one(rec: Dict[str, Any]) -> None:
        nonlocal processed_since_save, ckpt

        sid = record_sample_id(rec)
        async with sem:
            try:
                out = await generate_one(client, cfg, gen_cfg, job, rec)
                append_jsonl(job.output_path, out)
                ok = bool(out.get("success", False))
                ckpt.processed_ids[sid] = ok
                ckpt.processed_count = len(ckpt.processed_ids)
                if not ok:
                    ckpt.errors.append({"sample_id": sid, "error": out.get("error") or out.get("violations"), "time_utc": utc_now_iso()})
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                append_jsonl(
                    job.output_path,
                    {
                        "sample_id": sid,
                        "task": job.task,
                        "success": False,
                        "error": err,
                        "timestamp_utc": utc_now_iso(),
                    },
                )
                ckpt.processed_ids[sid] = False
                ckpt.processed_count = len(ckpt.processed_ids)
                ckpt.errors.append({"sample_id": sid, "error": err, "time_utc": utc_now_iso()})

            processed_since_save += 1
            if pbar is not None:
                pbar.update(1)

            if processed_since_save >= job.save_every:
                ckpt_mgr.save(ckpt)
                processed_since_save = 0

    try:
        # Schedule tasks in a memory-safe way: gather in batches
        batch_size = max(job.concurrency * 4, 64)
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            await asyncio.gather(*[_worker_one(r) for r in chunk])
        # Final save
        ckpt_mgr.save(ckpt)
    finally:
        if pbar is not None:
            pbar.close()
        await client.close()

    logger.info("Teacher job finished: output=%s | checkpoint=%s", str(job.output_path), str(job.checkpoint_path))


# -----------------------------
# CLI
# -----------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARTPipeline teacher generation (distill + repair).")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config. If omitted, use defaults.")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotted overrides key=value.")

    p.add_argument("--materialize", action="store_true", help="Materialize runs/{exp_id}/ via config.materialize_run().")
    p.add_argument("--task", type=str, required=True, choices=["distill", "repair"], help="Task type.")

    # Backward-compatible aliases: --distill-input / --repair-input
    # Our run scripts historically used task-specific flags; we map them to the unified --input.
    p.add_argument(
        "--input",
        "--distill-input",
        "--repair-input",
        dest="input",
        type=str,
        default=None,
        help="Input JSONL path. If omitted, use run defaults.",
    )
    p.add_argument("--splits", type=str, default=None, help="Comma-separated splits filter (e.g., train,val,test).")
    p.add_argument("--sources", type=str, default=None, help="Comma-separated sources filter (e.g., D1,D2).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap the number of samples (deterministic).")

    p.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests.")
    p.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval by processed samples.")
    p.add_argument("--no-resume", action="store_true", help="Disable resume/checkpoint loading.")
    p.add_argument("--reprocess-failed", action="store_true", help="Re-run samples with success=false in existing output.")

    # Quality control
    p.add_argument("--format-fix-rounds", type=int, default=2, help="Number of correction rounds for format/label issues.")
    p.add_argument("--no-sentence-check", action="store_true", help="Disable sentence range enforcement.")
    p.add_argument("--min-sentences", type=int, default=2, help="Min sentences for analysis.")
    p.add_argument("--max-sentences", type=int, default=4, help="Max sentences for analysis.")
    p.add_argument("--allow-quoting", action="store_true", help="Allow analysis to quote long prompt substrings.")
    p.add_argument("--quote-min-chars", type=int, default=40, help="Prompt quote detection threshold.")
    return p.parse_args()


def default_input_for_task(cfg: DARTPipelineConfig, task: str) -> Path:
    """
    Determine default input paths aligned with our pipeline outputs.
    """
    run_dir = Path(cfg.derived.get("paths", {}).get("run_dir", cfg.paths.runs_dir() / (cfg.run.exp_id or "run")))
    if task == "distill":
        # From datasets export
        cand = run_dir / "datasets" / "primary" / "train.jsonl"
        return cand
    # repair
    # From audit export (to be produced later). Provide a conventional location.
    cand = run_dir / "repair_pool" / "regression_pool.jsonl"
    return cand

def _auto_export_normalized_datasets(cfg: DARTPipelineConfig, run_dir: Path, *, include_external: bool = True) -> None:
    """Best-effort auto-export of normalized datasets to runs/{exp_id}/datasets/.

    This is a pragmatic safeguard so the standard pipeline can start from:
      python -m model.config --materialize ...
      python -m model.teacher_generate --materialize --task distill ...
    without requiring an explicit datasets.py step.
    """
    try:
        from .datasets import DatasetManager  # type: ignore
    except Exception:  # pragma: no cover
        from datasets import DatasetManager  # type: ignore

    dm = DatasetManager(cfg, log=logger)

    # Primary (required)
    primary_all = dm.load_primary()
    primary_splits = dm.split_primary(primary_all)

    # External (optional; best-effort)
    external_sets = None
    if include_external:
        try:
            external_sets = dm.load_all_external()
        except Exception as e:
            logger.warning("Auto-export: failed to load external suites; continuing with primary only. Error=%s", str(e))

    export_root = run_dir / "datasets"
    manifest = dm.export_normalized(primary_splits, external_sets, export_root=export_root, overwrite=True)

    # Record what we exported for reproducibility.
    try:
        cfg_dir = Path(cfg.derived.get("paths", {}).get("configs_dir", run_dir / "configs"))
        _safe_mkdir(cfg_dir)
        _json_dump_file(manifest, cfg_dir / "datasets.auto_export_manifest.json")
    except Exception:
        pass

def main() -> None:
    args = _cli()

    # Load config
    if args.config:
        cfg = DARTPipelineConfig.from_dict(load_config_file(Path(args.config)))
    else:
        cfg = build_default_config()

    cfg = apply_overrides(cfg, args.override)

    # Materialize run directories (recommended)
    if args.materialize:
        cfg.materialize_run(create_dirs=True)

    # Setup logging to run logs if available
    logs_dir = None
    if isinstance(cfg.derived, dict) and "paths" in cfg.derived:
        logs_dir = Path(cfg.derived["paths"]["logs_dir"])
    else:
        logs_dir = Path(cfg.paths.root()) / "runs" / (cfg.run.exp_id or "run") / "logs"
    configure_logging(logs_dir, level=logging.INFO, console=True)

    task = args.task

    # Determine paths
    run_dir = Path(cfg.derived.get("paths", {}).get("run_dir", cfg.paths.runs_dir() / (cfg.run.exp_id or "run")))
    out_dir = run_dir / "teacher_outputs" / task
    _safe_mkdir(out_dir)

    input_path = Path(args.input).expanduser().resolve() if args.input else default_input_for_task(cfg, task)

    # Pipeline guardrail: if the default distill input is missing, auto-export normalized datasets.
    # (This avoids a common failure mode where users run teacher_generate before datasets.py.)
    if (not input_path.exists()) and (task == "distill") and (args.input is None):
        logger.warning(
            "Default distill input not found: %s. Auto-exporting normalized datasets to %s ...",
            str(input_path),
            str(run_dir / "datasets"),
        )
        _auto_export_normalized_datasets(cfg, run_dir, include_external=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {input_path}. "
            "If you are running the standard pipeline, ensure normalized datasets exist under runs/{exp_id}/datasets/ "
            "(e.g., run model.datasets --materialize --export-normalized --include-external)."
        )
    output_path = out_dir / "outputs.jsonl"
    checkpoint_path = out_dir / "checkpoint.json"
    job_snapshot_path = out_dir / "job.snapshot.json"

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    job = TeacherJobConfig(
        task=task,  # type: ignore[arg-type]
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        job_snapshot_path=job_snapshot_path,
        splits=splits,
        sources=sources,
        max_samples=args.max_samples,
        concurrency=args.concurrency or cfg.teacher.max_concurrency,
        save_every=args.save_every,
        resume=not args.no_resume,
        reprocess_failed=args.reprocess_failed,
        format_fix_rounds=args.format_fix_rounds,
        enforce_sentence_range=not args.no_sentence_check,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        avoid_prompt_quoting=not args.allow_quoting,
        quote_min_chars=args.quote_min_chars,
    )

    # Run async
    asyncio.run(run_teacher_job(cfg, job))


if __name__ == "__main__":
    main()