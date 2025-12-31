# model/datasets.py
"""
Dataset loading and normalization for the DARTPipeline (ACL-level reproducibility).

This module provides:
- A canonical normalized sample schema shared across all pipeline stages:
    {sample_id, source, prompt, gold_label, split, meta}
- Robust loaders for:
    - 8 primary benchmarks (D1–D4, N1–N4) stored as pickles under benchmark_suite/
      (each pkl contains [diff_samples, equal_samples]; each item typically [question, answer, scenario_id])
    - External audit/eval suites under dataset/:
        * BOLD (prompts/*.json + wikipedia/*.json)
        * HolisticBias (descriptors.json, nouns.json, sentence_templates.json, standalone_noun_phrases.json)
        * RealToxicityPrompts (prompts.jsonl)
        * HateCheck (test_suite_cases.csv + optional annotations)
- Deterministic stratified splits by (benchmark_id, condition) for primary benchmarks
- Deterministic sampling for large external suites
- Dataset cards (JSON) that document file paths, schema, sizes, and key distributions
- Export of normalized datasets to JSONL for downstream training/inference/audit

Assumed repository layout (project root):
  project_root/
    model/
      config.py
      datasets.py
      ...
    dataset/
      bold/
      hatecheck/
      holisticbias/
      realtoxicityprompts/
    benchmark_suite/
      D1_1k.pkl ... N4_1k.pkl
    runs/
      {exp_id}/...

Recommended invocation:
  python -m model.datasets --materialize --export-normalized --include-external

Notes:
- Comments are intentionally English for publication-grade code; the user-facing discussion can be Chinese.
- This file does not depend on the old pipeline modules. It is a fresh, complete implementation aligned with DARTPipeline.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

# Optional acceleration: pandas is not required.
try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# Import config with both package and script compatibility.
try:
    from .config import (
        DARTPipelineConfig,
        BenchmarkSourceConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        _read_json,  # type: ignore
        _json_dump_file,  # type: ignore
        _safe_mkdir,  # type: ignore
    )
except Exception:  # pragma: no cover
    from config import (  # type: ignore
        DARTPipelineConfig,
        BenchmarkSourceConfig,
        build_default_config,
        load_config_file,
        apply_overrides,
        configure_logging,
        _read_json,
        _json_dump_file,
        _safe_mkdir,
    )


logger = logging.getLogger("dartpipeline.datasets")


# -----------------------------
# Normalized schema
# -----------------------------

GoldLabel = Literal["YES", "NO"]


@dataclass
class NormalizedSample:
    """
    Canonical normalized sample used across DARTPipeline.

    - gold_label is optional (None for external suites without difference-awareness labels).
    - split is optional; external suites default to 'test' for evaluation.
    - meta stores dataset-specific fields for slicing and reporting.
    """
    sample_id: str
    source: str
    prompt: str
    gold_label: Optional[GoldLabel] = None
    split: Optional[Literal["train", "val", "test"]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "prompt": self.prompt,
            "gold_label": self.gold_label,
            "split": self.split,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizedSample":
        return NormalizedSample(
            sample_id=str(d["sample_id"]),
            source=str(d["source"]),
            prompt=str(d["prompt"]),
            gold_label=d.get("gold_label"),
            split=d.get("split"),
            meta=dict(d.get("meta") or {}),
        )


# -----------------------------
# File & hashing utilities
# -----------------------------

def resolve_path(project_root: Path, p: str) -> Path:
    """
    Resolve a configured path that may be absolute or project-root-relative.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root / pp).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 of a file in streaming mode.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_metadata(path: Path, compute_hash: bool = True) -> Dict[str, Any]:
    """
    Collect stable file metadata for reproducibility.
    """
    st = path.stat()
    meta = {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "modified_time": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "extension": path.suffix.lower(),
    }
    if compute_hash:
        meta["sha256"] = sha256_file(path)
    return meta


def stable_text_hash(text: str, n: int = 12) -> str:
    """
    Stable short hash of a string (used for sample_id when raw IDs are missing).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def ensure_unique_sample_ids(samples: Sequence[NormalizedSample]) -> None:
    """
    Validate uniqueness of sample_id within a dataset.
    """
    seen = set()
    dupes = []
    for s in samples:
        if s.sample_id in seen:
            dupes.append(s.sample_id)
        else:
            seen.add(s.sample_id)
    if dupes:
        raise ValueError(f"Duplicate sample_id detected (showing up to 10): {dupes[:10]}")


# -------------------------
# Overview-based discovery
# -------------------------

def _is_inventory_overview(obj: Any) -> bool:
    return isinstance(obj, dict) and "files" in obj and isinstance(obj.get("files"), list)


def discover_paths_from_inventory_overview(
    overview_obj: Dict[str, Any],
) -> Dict[str, str]:
    """
    Parse the inventory-style dataset_overview_summary.json used in your local setup,
    which contains a list of files with 'relative_path'.

    Returns a mapping: basename -> relative_path
      e.g., "D1_1k.pkl" -> "benchmark_suite/D1_1k.pkl"
    """
    out: Dict[str, str] = {}
    for rec in overview_obj.get("files", []):
        try:
            rel = rec["metadata"]["relative_path"]
            bn = Path(rel).name
            out[bn] = rel
        except Exception:
            continue
    return out


# -----------------------------
# Primary benchmarks loader
# -----------------------------

def load_primary_benchmark_pkl(path: Path) -> Tuple[List[Tuple[Any, Any, Any]], List[Tuple[Any, Any, Any]]]:
    """
    Load a primary benchmark pickle.

    Expected canonical format:
      data = [diff_raw, equal_raw]
      diff_raw: list of items, each item like [question, answer, scenario_id] (or tuple)
      equal_raw: same

    This loader is robust to minor variations:
      - data may be a tuple (diff_raw, equal_raw)
      - data may be dict with keys like "diff"/"equal" or 0/1
      - item may be dict with keys {question/answer/scenario_id} or similar
    """
    with path.open("rb") as f:
        data = pickle.load(f)

    diff_raw = None
    equal_raw = None

    if isinstance(data, (list, tuple)) and len(data) >= 2:
        diff_raw, equal_raw = data[0], data[1]
    elif isinstance(data, dict):
        # Try common keys
        for dk in ["diff", "DIFF", "difference", "requires_difference", "requires_difference_awareness", 0, "0"]:
            if dk in data:
                diff_raw = data[dk]
                break
        for ek in ["equal", "EQUAL", "same", "no_difference", 1, "1"]:
            if ek in data:
                equal_raw = data[ek]
                break
        # If still missing, try first two values
        if diff_raw is None or equal_raw is None:
            vals = list(data.values())
            if len(vals) >= 2:
                diff_raw = diff_raw if diff_raw is not None else vals[0]
                equal_raw = equal_raw if equal_raw is not None else vals[1]

    if diff_raw is None or equal_raw is None:
        raise ValueError(f"Unrecognized primary benchmark pickle structure: {path}")

    # Normalize to list
    if not isinstance(diff_raw, list):
        diff_raw = list(diff_raw)
    if not isinstance(equal_raw, list):
        equal_raw = list(equal_raw)

    def _norm_item(it: Any) -> Tuple[Any, Any, Any]:
        if isinstance(it, (list, tuple)) and len(it) >= 3:
            return it[0], it[1], it[2]
        if isinstance(it, dict):
            q = it.get("question") or it.get("prompt") or it.get("text")
            a = it.get("answer") or it.get("label") or it.get("gold") or it.get("correct_answer")
            sid = it.get("scenario_id") or it.get("id") or it.get("uid") or it.get("scenario")
            return q, a, sid
        # Last resort: treat as single question string with unknown fields
        return it, None, None

    diff_norm = [_norm_item(it) for it in diff_raw]
    equal_norm = [_norm_item(it) for it in equal_raw]
    return diff_norm, equal_norm


def normalize_primary_samples(
    benchmark_id: str,
    diff_raw: List[Tuple[Any, Any, Any]],
    equal_raw: List[Tuple[Any, Any, Any]],
) -> List[NormalizedSample]:
    """
    Convert raw primary benchmark records to normalized samples.

    IMPORTANT: Some primary benchmarks (notably D3 in our local inventory) contain
    duplicated scenario_id values, and occasionally even duplicated (scenario_id, prompt)
    pairs. We therefore construct sample_id to be *deterministically unique*:

      base_id = f"{benchmark_id}:{condition}:{scenario_id_or_hash}:{prompt_hash8}"
      if base_id repeats in-order, append "__{k}" where k is the 1-based repeat index.

    This preserves human-readable scenario identifiers while guaranteeing uniqueness
    required by downstream pairing/audit logic.
    """
    out: List[NormalizedSample] = []
    id_counts: Dict[str, int] = {}

    def _make_sample(condition: str, rec: Tuple[Any, Any, Any]) -> NormalizedSample:
        question, answer, scenario_id = rec
        prompt = "" if question is None else str(question)
        scenario_id_str = "" if scenario_id is None else str(scenario_id)

        gold_label: GoldLabel = "YES" if condition == "diff" else "NO"
        label_int = 1 if gold_label == "YES" else 0

        # Deterministic uniqueness:
        # - keep scenario_id when available for readability
        # - add a short prompt hash to avoid collisions when scenario_id repeats
        # - add an in-order suffix if even (scenario_id, prompt) repeats
        prompt_hash8 = stable_text_hash(prompt, n=8)
        if scenario_id_str != "":
            scenario_part = scenario_id_str
        else:
            scenario_part = f"h{stable_text_hash(prompt, n=12)}"

        base_id = f"{benchmark_id}:{condition}:{scenario_part}:{prompt_hash8}"
        k = id_counts.get(base_id, 0)
        id_counts[base_id] = k + 1
        sample_id = base_id if k == 0 else f"{base_id}__{k}"

        meta = {
            "task_type": "difference_awareness_classification",
            "benchmark_id": benchmark_id,
            "condition": condition,
            "requires_difference_awareness": (condition == "diff"),
            "label_int": label_int,
            "scenario_id": scenario_id_str,
            "scenario_id_raw": scenario_id_str,
            "prompt_hash8": prompt_hash8,
            "sample_id_base": base_id,
            "answer": int(answer) if isinstance(answer, (int, float, str)) and str(answer).strip().isdigit() else answer,
        }

        return NormalizedSample(
            sample_id=sample_id,
            source=benchmark_id,
            prompt=prompt,
            gold_label=gold_label,
            split=None,
            meta=meta,
        )

    out.extend([_make_sample("diff", rec) for rec in diff_raw])
    out.extend([_make_sample("equal", rec) for rec in equal_raw])
    ensure_unique_sample_ids(out)
    return out


def stratified_split_primary(
    samples: List[NormalizedSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[NormalizedSample]]:
    """
    Stratified split by (benchmark_id, condition) to match the original pipeline.

    This replicates the structure used in your earlier code:
      group_key = (benchmark_id, condition)
      random.shuffle within group
      allocate by ratios
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[NormalizedSample]] = {}
    for s in samples:
        bid = str(s.meta.get("benchmark_id", s.source))
        cond = str(s.meta.get("condition", "unknown"))
        key = (bid, cond)
        groups.setdefault(key, []).append(s)

    out = {"train": [], "val": [], "test": []}  # type: ignore[dict-item]
    for _, g in groups.items():
        rng.shuffle(g)
        n = len(g)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder to test
        train_g = g[:n_train]
        val_g = g[n_train:n_train + n_val]
        test_g = g[n_train + n_val:]

        for s in train_g:
            s.split = "train"
        for s in val_g:
            s.split = "val"
        for s in test_g:
            s.split = "test"

        out["train"].extend(train_g)
        out["val"].extend(val_g)
        out["test"].extend(test_g)

    # Shuffle splits to avoid grouped ordering
    for split in ["train", "val", "test"]:
        rng.shuffle(out[split])

    return out


# -----------------------------
# External suite loaders
# -----------------------------

def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL file.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e


def reservoir_sample_iter(
    it: Iterator[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Reservoir sampling for streaming data.
    """
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    for i, obj in enumerate(it):
        if i < k:
            sample.append(obj)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = obj
    return sample


def load_realtoxicityprompts(
    jsonl_path: Path,
    max_samples: Optional[int],
    seed: int,
) -> List[NormalizedSample]:
    """
    Load RealToxicityPrompts from prompts.jsonl.
    Your local file appears to have columns: filename, begin, end, challenging, prompt, continuation
    (and may include additional toxicity fields in some variants).
    """
    it = iter_jsonl(jsonl_path)
    if max_samples is not None:
        objs = reservoir_sample_iter(it, max_samples, seed=seed)
    else:
        objs = list(it)

    out: List[NormalizedSample] = []
    for idx, obj in enumerate(objs):
        prompt = obj.get("text") or obj.get("prompt") or obj.get("input") or ""
        prompt = str(prompt)

        sid = obj.get("id") or obj.get("prompt_id") or obj.get("filename") or f"{idx}"
        sid = str(sid)
        sample_id = f"RealToxicityPrompts:{sid}"

        meta = dict(obj)
        meta.update(
            {
                "suite": "RealToxicityPrompts",
                "task_type": "continuation",
                "source_file": str(jsonl_path),
            }
        )

        out.append(
            NormalizedSample(
                sample_id=sample_id,
                source="RealToxicityPrompts",
                prompt=prompt,
                gold_label=None,
                split="test",
                meta=meta,
            )
        )

    ensure_unique_sample_ids(out)
    return out


def _extract_strings_from_obj(obj: Any, max_items: int = 10) -> List[str]:
    """
    Collect string leaves from nested structures for robust prompt extraction.
    """
    out: List[str] = []

    def _walk(x: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
                if len(out) >= max_items:
                    return
        if isinstance(x, list):
            for v in x:
                _walk(v)
                if len(out) >= max_items:
                    return

    _walk(obj)
    return out


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    """
    Extract first N sentences for wiki context. Conservative splitting.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Simple sentence segmentation based on period/question/exclamation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sentences]).strip()


def load_bold(
    bold_root: Path,
    max_samples: Optional[int],
    seed: int,
    include_wiki_context: bool = False,
    wiki_sentences: int = 2,
) -> List[NormalizedSample]:
    """
    Load BOLD prompts from your local structure:
      dataset/bold/prompts/*_prompt.json
      dataset/bold/wikipedia/*_wiki.json

    Each prompt json is typically:
      { group_name: { entity_name: <prompt or structure> , ... }, ... }

    We implement robust extraction:
      - If entity value is string -> prompt
      - If dict/list -> extract strings from common keys or string leaves
    """
    prompts_dir = bold_root / "prompts"
    wiki_dir = bold_root / "wikipedia"

    if not prompts_dir.exists():
        raise FileNotFoundError(f"BOLD prompts directory not found: {prompts_dir}")

    prompt_files = sorted(prompts_dir.glob("*_prompt.json"))
    if not prompt_files:
        raise FileNotFoundError(f"No BOLD prompt files matching '*_prompt.json' under: {prompts_dir}")

    # Load wiki files if available
    wiki_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for wf in sorted(wiki_dir.glob("*_wiki.json")) if wiki_dir.exists() else []:
        dimension = wf.stem.replace("_wiki", "")
        try:
            wiki_maps[dimension] = _json_load(wf)
        except Exception:
            continue

    rng = random.Random(seed)
    all_records: List[Tuple[str, str, str, str, Optional[str]]] = []
    # record: (dimension, group, entity, prompt_text, wiki_text)
    for pf in prompt_files:
        dimension = pf.stem.replace("_prompt", "")
        data = _json_load(pf)
        if not isinstance(data, dict):
            continue
        for group_name, group_obj in data.items():
            if isinstance(group_obj, dict):
                items = group_obj.items()
            else:
                # If group_obj is list, treat as entityless prompts
                items = [(f"idx_{i}", v) for i, v in enumerate(group_obj if isinstance(group_obj, list) else [group_obj])]
            for entity, entity_obj in items:
                prompts: List[str] = []
                if isinstance(entity_obj, str):
                    prompts = [entity_obj]
                elif isinstance(entity_obj, dict):
                    # Prefer explicit keys
                    for key in ["prompt", "text", "completion_prompt", "prefix", "input"]:
                        if key in entity_obj and isinstance(entity_obj[key], str):
                            prompts = [entity_obj[key]]
                            break
                    if not prompts:
                        if "prompts" in entity_obj and isinstance(entity_obj["prompts"], list):
                            prompts = [str(x) for x in entity_obj["prompts"] if isinstance(x, (str, int, float))]
                elif isinstance(entity_obj, list):
                    prompts = [str(x) for x in entity_obj if isinstance(x, (str, int, float))]
                if not prompts:
                    prompts = _extract_strings_from_obj(entity_obj, max_items=3)
                if not prompts:
                    continue

                wiki_text: Optional[str] = None
                if include_wiki_context:
                    wdim = wiki_maps.get(dimension)
                    if isinstance(wdim, dict):
                        wgroup = wdim.get(group_name)
                        if isinstance(wgroup, dict) and entity in wgroup:
                            wv = wgroup.get(entity)
                            if isinstance(wv, str):
                                wiki_text = _first_sentences(wv, max_sentences=wiki_sentences)
                            elif isinstance(wv, dict):
                                # try typical keys
                                for k in ["text", "wiki", "summary", "intro"]:
                                    if k in wv and isinstance(wv[k], str):
                                        wiki_text = _first_sentences(wv[k], max_sentences=wiki_sentences)
                                        break
                            elif isinstance(wv, list) and wv and isinstance(wv[0], str):
                                wiki_text = _first_sentences(wv[0], max_sentences=wiki_sentences)

                for ptxt in prompts:
                    all_records.append((dimension, str(group_name), str(entity), str(ptxt), wiki_text))

    if not all_records:
        raise ValueError(f"No BOLD prompts extracted from: {prompts_dir}")

    if max_samples is not None and len(all_records) > max_samples:
        rng.shuffle(all_records)
        all_records = all_records[:max_samples]

    out: List[NormalizedSample] = []
    id_counts: Dict[str, int] = {}

    for dimension, group, entity, prompt_text, wiki_text in all_records:
        base_prompt = prompt_text.strip()
        if include_wiki_context and wiki_text:
            prompt = f"{wiki_text}\n\n{base_prompt}"
        else:
            prompt = base_prompt

        # Base id keeps human-readable taxonomy + prompt hash; may still collide if dataset has duplicates.
        base_id = f"BOLD:{dimension}:{group}:{entity}:{stable_text_hash(prompt, n=10)}"
        k = id_counts.get(base_id, 0)
        id_counts[base_id] = k + 1
        sample_id = base_id if k == 0 else f"{base_id}__{k}"

        meta = {
            "suite": "BOLD",
            "task_type": "continuation",
            "dimension": dimension,
            "group": group,
            "entity": entity,
            "include_wiki_context": include_wiki_context,
            "sample_id_base": base_id,
            "dup_index": k,
        }
        if wiki_text:
            meta["wiki_snippet"] = wiki_text

        out.append(
            NormalizedSample(
                sample_id=sample_id,
                source="BOLD",
                prompt=prompt,
                gold_label=None,
                split="test",
                meta=meta,
            )
        )

    ensure_unique_sample_ids(out)
    return out


def _pluralize_naive(noun_phrase: str) -> str:
    """
    Very naive pluralization; used only as fallback for HolisticBias.
    """
    w = noun_phrase.strip()
    if not w:
        return w
    # pluralize the last word
    parts = w.split()
    last = parts[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last_p = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        last_p = last + "es"
    else:
        last_p = last + "s"
    parts[-1] = last_p
    return " ".join(parts)


def _normalize_holisticbias_noun_phrase_item(item: Any) -> Dict[str, Any]:
    """
    Normalize an item from standalone_noun_phrases into a dict with noun_phrase/plural_noun_phrase if possible.
    """
    if isinstance(item, str):
        return {"noun_phrase": item, "plural_noun_phrase": _pluralize_naive(item)}
    if isinstance(item, dict):
        out = dict(item)
        if "noun_phrase" not in out:
            # common alternative keys
            for k in ["phrase", "text", "noun"]:
                if k in out and isinstance(out[k], str):
                    out["noun_phrase"] = out[k]
                    break
        if "plural_noun_phrase" not in out and "noun_phrase" in out and isinstance(out["noun_phrase"], str):
            out["plural_noun_phrase"] = _pluralize_naive(out["noun_phrase"])
        return out
    return {"noun_phrase": str(item), "plural_noun_phrase": _pluralize_naive(str(item))}


def load_holisticbias(
    hb_root: Path,
    max_samples: Optional[int],
    seed: int,
    per_category_cap: Optional[int] = None,
) -> List[NormalizedSample]:
    """
    Generate HolisticBias prompts from combinatorial vocab + templates.

    Local files:
      - descriptors.json: {category: [descriptor, ...], ...}
      - nouns.json: {female: [...], male: [...], neutral: [...]}
      - sentence_templates.json: {template_string: {constraints...}, ...}
      - standalone_noun_phrases.json: {category: [noun_phrase items], ...}

    Generation strategy (balanced, deterministic):
      - For each descriptor category:
          sample noun_phrases (prefer standalone_noun_phrases; fallback to descriptor×noun)
      - For each noun_phrase, sample a template (keys of sentence_templates.json)
      - Format template using {noun_phrase} or {plural_noun_phrase}
      - Cap per category to keep coverage balanced if max_samples is small.

    Output: normalized samples with meta carrying category, descriptor/noun if available, template, etc.
    """
    desc_path = hb_root / "descriptors.json"
    nouns_path = hb_root / "nouns.json"
    templates_path = hb_root / "sentence_templates.json"
    standalone_path = hb_root / "standalone_noun_phrases.json"

    for p in [desc_path, nouns_path, templates_path]:
        if not p.exists():
            raise FileNotFoundError(f"HolisticBias required file missing: {p}")

    descriptors = _json_load(desc_path)
    nouns = _json_load(nouns_path)
    templates_obj = _json_load(templates_path)
    standalone = _json_load(standalone_path) if standalone_path.exists() else {}

    if not isinstance(descriptors, dict) or not isinstance(nouns, dict) or not isinstance(templates_obj, dict):
        raise ValueError("HolisticBias JSON structure unexpected (expected dicts).")

    templates = list(templates_obj.keys())
    if not templates:
        raise ValueError(f"No templates found in: {templates_path}")

    rng = random.Random(seed)

    # Prepare noun pools by group
    noun_groups: Dict[str, List[str]] = {}
    for g, lst in nouns.items():
        if isinstance(lst, list):
            noun_groups[str(g)] = [str(x) for x in lst if isinstance(x, (str, int, float))]

    # Determine categories
    categories = [str(c) for c in descriptors.keys()]
    categories.sort()

    # Determine default per-category cap
    if max_samples is not None and per_category_cap is None and categories:
        per_category_cap = max(1, max_samples // len(categories))

    all_samples: List[NormalizedSample] = []
    for cat in categories:
        # Get prebuilt noun phrases for this category if available
        noun_phrase_items = []
        if isinstance(standalone, dict) and cat in standalone and isinstance(standalone[cat], list):
            noun_phrase_items = [_normalize_holisticbias_noun_phrase_item(x) for x in standalone[cat]]

        # Fallback: synthesize from descriptor×noun with light sampling
        if not noun_phrase_items:
            desc_list = descriptors.get(cat, [])
            if not isinstance(desc_list, list) or not desc_list:
                continue
            # Sample some descriptors and nouns for combinatorial coverage
            # Use a conservative cap to avoid explosion
            desc_sample = [str(x) for x in desc_list if isinstance(x, (str, int, float))]
            rng.shuffle(desc_sample)
            desc_sample = desc_sample[: min(len(desc_sample), 50)]

            noun_candidates: List[Tuple[str, str]] = []  # (noun_group, noun)
            for ng, nlst in noun_groups.items():
                for n in nlst[:50]:
                    noun_candidates.append((ng, n))
            rng.shuffle(noun_candidates)
            noun_candidates = noun_candidates[: min(len(noun_candidates), 100)]

            for dsc in desc_sample:
                for ng, n in noun_candidates:
                    phrase = f"{dsc} {n}".strip()
                    noun_phrase_items.append(
                        {
                            "noun_phrase": phrase,
                            "plural_noun_phrase": _pluralize_naive(phrase),
                            "descriptor": dsc,
                            "noun": n,
                            "noun_group": ng,
                            "category": cat,
                        }
                    )

        # Shuffle for sampling
        rng.shuffle(noun_phrase_items)

        # Category cap (coverage balance)
        cap = per_category_cap if per_category_cap is not None else len(noun_phrase_items)
        noun_phrase_items = noun_phrase_items[: min(len(noun_phrase_items), cap)]

        for item in noun_phrase_items:
            noun_phrase = str(item.get("noun_phrase", "")).strip()
            plural_noun_phrase = str(item.get("plural_noun_phrase", "")).strip() or _pluralize_naive(noun_phrase)
            if not noun_phrase:
                continue

            template = rng.choice(templates)
            # Choose which placeholder is required by template
            if "{plural_noun_phrase}" in template:
                prompt = template.replace("{plural_noun_phrase}", plural_noun_phrase)
                prompt = prompt.replace("{noun_phrase}", noun_phrase)
            else:
                prompt = template.replace("{noun_phrase}", noun_phrase)
                prompt = prompt.replace("{plural_noun_phrase}", plural_noun_phrase)

            # sample_id includes hash of prompt for stability
            sample_id = f"HolisticBias:{cat}:{stable_text_hash(prompt, n=10)}"

            meta = {
                "suite": "HolisticBias",
                "task_type": "continuation",
                "category": cat,
                "noun_phrase": noun_phrase,
                "plural_noun_phrase": plural_noun_phrase,
                "template": template,
            }
            # Preserve additional info if present
            for k in ["descriptor", "noun", "noun_group"]:
                if k in item:
                    meta[k] = item[k]

            all_samples.append(
                NormalizedSample(
                    sample_id=sample_id,
                    source="HolisticBias",
                    prompt=prompt,
                    gold_label=None,
                    split="test",
                    meta=meta,
                )
            )

    if not all_samples:
        raise ValueError(f"No HolisticBias prompts generated from: {hb_root}")

    # Global cap (deterministic)
    if max_samples is not None and len(all_samples) > max_samples:
        rng.shuffle(all_samples)
        all_samples = all_samples[:max_samples]

    ensure_unique_sample_ids(all_samples)
    return all_samples


def load_hatecheck_cases(
    cases_csv: Path,
    annotations_csv: Optional[Path] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[NormalizedSample]:
    """
    Load HateCheck cases from CSV.

    Your local file test_suite_cases.csv contains columns such as:
      functionality, case_id, test_case, label_gold, target_ident, direction, focus_words, ...

    We treat test_case as the "user message" and expect inference.py to format a chat response.

    If annotations are provided, we merge any additional fields by case_id when possible.
    """
    if not cases_csv.exists():
        raise FileNotFoundError(f"HateCheck cases CSV not found: {cases_csv}")

    rng = random.Random(seed)

    # Read cases
    rows: List[Dict[str, Any]] = []
    if pd is not None:
        try:
            df = pd.read_csv(cases_csv)
            rows = df.to_dict(orient="records")
        except Exception:
            rows = []
    if not rows:
        with cases_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

    # Optional annotations merge
    ann_map: Dict[str, Dict[str, Any]] = {}
    if annotations_csv and annotations_csv.exists():
        ann_rows: List[Dict[str, Any]] = []
        if pd is not None:
            try:
                df_ann = pd.read_csv(annotations_csv)
                ann_rows = df_ann.to_dict(orient="records")
            except Exception:
                ann_rows = []
        if not ann_rows:
            with annotations_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ann_rows.append(r)

        # Index by case_id if present
        for r in ann_rows:
            cid = r.get("case_id") or r.get("CaseID") or r.get("id")
            if cid is None:
                continue
            cid = str(cid)
            # If multiple annotations per case exist, we store one representative and counts.
            if cid not in ann_map:
                ann_map[cid] = dict(r)
                ann_map[cid]["_annotation_rows"] = 1
            else:
                ann_map[cid]["_annotation_rows"] = ann_map[cid].get("_annotation_rows", 1) + 1

    # Deterministic cap
    if max_samples is not None and len(rows) > max_samples:
        rng.shuffle(rows)
        rows = rows[:max_samples]

    out: List[NormalizedSample] = []
    for idx, r in enumerate(rows):
        cid = r.get("case_id") or r.get("CaseID") or r.get("id") or str(idx)
        cid = str(cid)
        test_case = r.get("test_case") or r.get("text") or r.get("case") or ""
        test_case = str(test_case)

        sample_id = f"HateCheck:{cid}"
        meta = dict(r)
        meta.update(
            {
                "suite": "HateCheck",
                "task_type": "dialogue_safety",
                "source_file": str(cases_csv),
            }
        )
        # Merge annotations
        if cid in ann_map:
            meta["annotations"] = ann_map[cid]

        out.append(
            NormalizedSample(
                sample_id=sample_id,
                source="HateCheck",
                prompt=test_case,
                gold_label=None,
                split="test",
                meta=meta,
            )
        )

    ensure_unique_sample_ids(out)
    return out


# -----------------------------
# Dataset cards & export
# -----------------------------

@dataclass
class DatasetCard:
    """
    A compact dataset card for paper-quality documentation.
    """
    source_id: str
    display_name: str
    resolved_paths: List[Dict[str, Any]]
    normalized_schema: Dict[str, str]
    counts: Dict[str, Any]
    meta_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _value_counts(items: Iterable[Any], max_items: int = 20) -> List[Tuple[Any, int]]:
    c: Dict[Any, int] = {}
    for x in items:
        c[x] = c.get(x, 0) + 1
    return sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:max_items]


def summarize_samples_for_card(samples: List[NormalizedSample], slice_keys: List[str]) -> Dict[str, Any]:
    """
    Compute dataset-level statistics useful for ACL reporting.
    """
    counts: Dict[str, Any] = {}
    counts["n_total"] = len(samples)
    counts["n_by_split"] = dict(_value_counts([s.split for s in samples], max_items=10))
    counts["n_with_gold_label"] = sum(1 for s in samples if s.gold_label is not None)
    if counts["n_with_gold_label"] > 0:
        counts["n_by_gold_label"] = dict(
            _value_counts([s.gold_label for s in samples if s.gold_label is not None], max_items=10)
        )

    # Meta key summary
    meta_keys = set()
    for s in samples:
        meta_keys.update((s.meta or {}).keys())
    meta_summary: Dict[str, Any] = {"meta_keys_present": sorted(list(meta_keys))}

    # Slice distributions for selected keys
    for k in slice_keys:
        vals = []
        for s in samples:
            if k in (s.meta or {}):
                vals.append(s.meta.get(k))
        if vals:
            meta_summary[f"top_{k}"] = _value_counts(vals, max_items=20)

    return {"counts": counts, "meta_summary": meta_summary}


def _json_default(o: Any) -> Any:
    """
    Make common non-standard scalar types JSON-serializable.
    - numpy scalars (e.g., int64/float32): have .item()
    - numpy arrays: have .tolist()
    - Path: stringify
    Fallback: stringify.
    """
    # numpy / pandas scalars
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass

    # numpy arrays
    if hasattr(o, "tolist") and callable(getattr(o, "tolist")):
        try:
            return o.tolist()
        except Exception:
            pass

    if isinstance(o, Path):
        return str(o)

    return str(o)


def export_jsonl(samples: List[NormalizedSample], path: Path) -> None:
    """
    Export normalized samples to JSONL.
    """
    _safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False, default=_json_default) + "\n")


def load_jsonl_samples(path: Path) -> List[NormalizedSample]:
    """
    Load normalized samples from JSONL.
    """
    out: List[NormalizedSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(NormalizedSample.from_dict(json.loads(line)))
            except Exception as e:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {e}") from e
    ensure_unique_sample_ids(out)
    return out


# -----------------------------
# DatasetManager: main entry
# -----------------------------

class DatasetManager:
    """
    End-to-end dataset manager:
    - resolve file paths from config + local directory structure
    - load + normalize primary benchmarks
    - load + normalize external suites
    - create stratified splits (primary)
    - write dataset cards + normalized exports (optional)
    """

    def __init__(self, cfg: DARTPipelineConfig, log: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.project_root = Path(cfg.paths.root())
        self.log = log or logger

        # Ensure defaults exist
        self.cfg.data.ensure_defaults(self.cfg.paths)

        # Attempt to adapt source paths to your local inventory file if present
        self._apply_inventory_overview_if_present()

    def _apply_inventory_overview_if_present(self) -> None:
        """
        Your dataset_overview_summary.json is an inventory with a file list.
        If present, use it to patch dataset source paths to the observed relative paths.
        """
        overview_path = self.cfg.paths.find_dataset_overview_summary()
        if overview_path is None or not overview_path.exists():
            return
        try:
            obj = _read_json(overview_path)
        except Exception:
            return
        if not _is_inventory_overview(obj):
            return

        inv_map = discover_paths_from_inventory_overview(obj)
        # Patch dataset paths by filename match
        for _, src in self.cfg.data.sources.items():
            bn = Path(src.path).name
            if bn in inv_map:
                src.path = inv_map[bn]
                ext = Path(src.path).suffix.lower().lstrip(".")
                if ext:
                    src.format = ext  # type: ignore[assignment]

        self.log.info("Patched dataset source paths from inventory overview: %s", str(overview_path))

    def _resolve_source_path_candidates(self, src: BenchmarkSourceConfig) -> List[Path]:
        """
        Build a list of candidate paths for a dataset source, using:
          - configured path
          - common local structure fallbacks
          - benchmark_suite/ fallback for primary
        """
        candidates: List[Path] = []
        candidates.append(resolve_path(self.project_root, src.path))

        # Primary benchmark fallback: benchmark_suite/{ID}_1k.pkl
        if src.source_id in self.cfg.data.primary_benchmark_ids:
            fn = f"{src.source_id}_1k.pkl"
            candidates.append(self.project_root / "benchmark_suite" / fn)
            candidates.append(self.project_root / "dataset" / fn)
            candidates.append(self.project_root / "dataset" / "benchmark_suite" / fn)

        # External suites: common root folders
        if src.source_id in ["BOLD", "HolisticBias", "RealToxicityPrompts", "HateCheck"]:
            if src.source_id == "BOLD":
                candidates.append(self.project_root / "dataset" / "bold")
            elif src.source_id == "HolisticBias":
                candidates.append(self.project_root / "dataset" / "holisticbias")
            elif src.source_id == "RealToxicityPrompts":
                candidates.append(self.project_root / "dataset" / "realtoxicityprompts" / "prompts.jsonl")
            elif src.source_id == "HateCheck":
                candidates.append(self.project_root / "dataset" / "hatecheck" / "test_suite_cases.csv")

        # De-duplicate
        uniq: List[Path] = []
        seen = set()
        for p in candidates:
            rp = str(p.resolve())
            if rp not in seen:
                uniq.append(p)
                seen.add(rp)
        return uniq

    def resolve_existing_path(self, src: BenchmarkSourceConfig) -> Path:
        """
        Resolve to the first existing candidate path; otherwise raise a detailed error.
        """
        candidates = self._resolve_source_path_candidates(src)
        for p in candidates:
            if p.exists():
                return p.resolve()
        msg = [
            f"Could not resolve dataset source '{src.source_id}' to an existing file/folder.",
            f"Configured path: {src.path}",
            "Tried candidates:",
        ]
        msg.extend([f"  - {str(p)}" for p in candidates])
        raise FileNotFoundError("\n".join(msg))

    def load_primary(self, benchmark_ids: Optional[List[str]] = None) -> List[NormalizedSample]:
        """
        Load and normalize all selected primary benchmarks (unsplit).
        """
        bids = benchmark_ids or self.cfg.data.primary_benchmark_ids
        all_samples: List[NormalizedSample] = []

        for bid in bids:
            if bid not in self.cfg.data.sources:
                raise KeyError(f"Primary benchmark '{bid}' not found in config.data.sources.")
            src = self.cfg.data.sources[bid]
            p = self.resolve_existing_path(src)
            diff_raw, equal_raw = load_primary_benchmark_pkl(p)
            samples = normalize_primary_samples(bid, diff_raw, equal_raw)
            for s in samples:
                s.meta["source_file"] = str(p)
            all_samples.extend(samples)
            self.log.info("Loaded primary %s: %d samples (diff+equal)", bid, len(samples))

        ensure_unique_sample_ids(all_samples)
        return all_samples

    def split_primary(self, samples: List[NormalizedSample]) -> Dict[str, List[NormalizedSample]]:
        """
        Create stratified train/val/test splits for primary benchmarks.
        """
        first = self.cfg.data.sources[self.cfg.data.primary_benchmark_ids[0]]
        tr, vr, ter = first.split_ratios
        splits = stratified_split_primary(samples, tr, vr, ter, seed=self.cfg.run.seed)
        self.log.info(
            "Primary splits: train=%d val=%d test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"])
        )
        return splits

    def load_external_suite(
        self,
        suite_id: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[NormalizedSample]:
        """
        Load one external suite by ID.
        """
        seed = self.cfg.run.seed if seed is None else seed

        if suite_id not in self.cfg.data.sources:
            raise KeyError(f"External suite '{suite_id}' not found in config.data.sources.")

        src = self.cfg.data.sources[suite_id]
        p = self.resolve_existing_path(src)

        # Determine max_samples from schema_hint if not explicitly provided
        if max_samples is None:
            ms = src.schema_hint.get("max_samples") if isinstance(src.schema_hint, dict) else None
            if isinstance(ms, int) and ms > 0:
                max_samples = ms
            else:
                default_caps = {
                    "BOLD": 1000,
                    "HolisticBias": 1000,
                    "RealToxicityPrompts": 1000,
                    "HateCheck": None,
                }
                max_samples = default_caps.get(suite_id, None)

        if suite_id == "RealToxicityPrompts":
            if p.is_dir():
                p = (p / "prompts.jsonl")
            samples = load_realtoxicityprompts(p, max_samples=max_samples, seed=seed)

        elif suite_id == "BOLD":
            bold_root = p if p.is_dir() else p.parent
            include_wiki = bool(src.schema_hint.get("include_wiki_context", False)) if isinstance(src.schema_hint, dict) else False
            wiki_sentences = int(src.schema_hint.get("wiki_sentences", 2)) if isinstance(src.schema_hint, dict) and "wiki_sentences" in src.schema_hint else 2
            samples = load_bold(
                bold_root,
                max_samples=max_samples,
                seed=seed,
                include_wiki_context=include_wiki,
                wiki_sentences=wiki_sentences,
            )

        elif suite_id == "HolisticBias":
            hb_root = p if p.is_dir() else p.parent
            per_cat_cap = None
            if isinstance(src.schema_hint, dict) and "per_category_cap" in src.schema_hint:
                try:
                    per_cat_cap = int(src.schema_hint["per_category_cap"])
                except Exception:
                    per_cat_cap = None
            samples = load_holisticbias(hb_root, max_samples=max_samples, seed=seed, per_category_cap=per_cat_cap)

        elif suite_id == "HateCheck":
            if p.is_dir():
                cases_csv = p / "test_suite_cases.csv"
            else:
                cases_csv = p
            annotations_csv = None
            ann_hint = None
            if isinstance(src.schema_hint, dict):
                ann_hint = src.schema_hint.get("annotations_csv")
            if ann_hint:
                annotations_csv = resolve_path(self.project_root, str(ann_hint))
            else:
                cand = cases_csv.parent / "test_suite_annotations.csv"
                if cand.exists():
                    annotations_csv = cand
            samples = load_hatecheck_cases(cases_csv, annotations_csv=annotations_csv, max_samples=max_samples, seed=seed)

        else:
            raise ValueError(f"Unsupported external suite id: {suite_id}")

        for s in samples:
            s.meta.setdefault("resolved_source_path", str(p))

        self.log.info("Loaded external suite %s: %d samples", suite_id, len(samples))
        return samples

    def load_all_external(
        self,
        suite_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[NormalizedSample]]:
        suite_ids = suite_ids or ["BOLD", "HolisticBias", "RealToxicityPrompts", "HateCheck"]
        out: Dict[str, List[NormalizedSample]] = {}
        for sid in suite_ids:
            out[sid] = self.load_external_suite(sid)
        return out

    def build_dataset_cards(
        self,
        primary_splits: Dict[str, List[NormalizedSample]],
        external_sets: Optional[Dict[str, List[NormalizedSample]]],
        compute_file_hash: bool = True,
    ) -> Dict[str, DatasetCard]:
        """
        Create dataset cards for each source.
        """
        cards: Dict[str, DatasetCard] = {}
        slice_keys = list(self.cfg.evaluation.default_slice_keys)

        # Primary cards: per benchmark_id
        for bid in self.cfg.data.primary_benchmark_ids:
            src = self.cfg.data.sources[bid]
            merged: List[NormalizedSample] = []
            for _, lst in primary_splits.items():
                for s in lst:
                    if s.source == bid:
                        merged.append(s)
            if not merged:
                continue

            try:
                p = self.resolve_existing_path(src)
                paths_meta = [file_metadata(p, compute_hash=compute_file_hash)]
            except Exception:
                paths_meta = [{"path": src.path, "exists": False}]

            summ = summarize_samples_for_card(merged, slice_keys=slice_keys)
            cards[bid] = DatasetCard(
                source_id=bid,
                display_name=src.display_name,
                resolved_paths=paths_meta,
                normalized_schema=self.cfg.data.normalized_schema,
                counts=summ["counts"],
                meta_summary=summ["meta_summary"],
            )

        # External cards
        if external_sets:
            for sid, samples in external_sets.items():
                if sid not in self.cfg.data.sources:
                    continue
                src = self.cfg.data.sources[sid]
                resolved_meta: List[Dict[str, Any]] = []
                try:
                    p = self.resolve_existing_path(src)
                    if p.is_file():
                        resolved_meta = [file_metadata(p, compute_hash=compute_file_hash)]
                    else:
                        key_files = []
                        if sid == "BOLD":
                            key_files = sorted((p / "prompts").glob("*_prompt.json")) + sorted((p / "wikipedia").glob("*_wiki.json"))
                        elif sid == "HolisticBias":
                            key_files = [
                                p / "descriptors.json",
                                p / "nouns.json",
                                p / "sentence_templates.json",
                                p / "standalone_noun_phrases.json",
                            ]
                        elif sid == "HateCheck":
                            key_files = [p / "test_suite_cases.csv", p / "test_suite_annotations.csv"]
                        for kf in key_files:
                            if kf.exists() and kf.is_file():
                                resolved_meta.append(file_metadata(kf, compute_hash=compute_file_hash))
                        if not resolved_meta:
                            resolved_meta = [{"path": str(p), "exists": True, "note": "folder source; key files not enumerated"}]
                except Exception:
                    resolved_meta = [{"path": src.path, "exists": False}]

                summ = summarize_samples_for_card(samples, slice_keys=slice_keys)
                cards[sid] = DatasetCard(
                    source_id=sid,
                    display_name=src.display_name,
                    resolved_paths=resolved_meta,
                    normalized_schema=self.cfg.data.normalized_schema,
                    counts=summ["counts"],
                    meta_summary=summ["meta_summary"],
                )

        return cards

    def export_normalized(
        self,
        primary_splits: Dict[str, List[NormalizedSample]],
        external_sets: Optional[Dict[str, List[NormalizedSample]]],
        export_root: Path,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Export normalized datasets to JSONL under export_root.

        Layout:
          export_root/
            primary/
              train.jsonl
              val.jsonl
              test.jsonl
            external/
              BOLD.jsonl
              HolisticBias.jsonl
              RealToxicityPrompts.jsonl
              HateCheck.jsonl
        """
        _safe_mkdir(export_root)
        manifest: Dict[str, Any] = {"export_root": str(export_root), "primary": {}, "external": {}}

        primary_dir = export_root / "primary"
        _safe_mkdir(primary_dir)
        for split, samples in primary_splits.items():
            out_path = primary_dir / f"{split}.jsonl"
            if out_path.exists() and not overwrite:
                self.log.warning("Skip export (exists): %s", str(out_path))
            else:
                export_jsonl(samples, out_path)
            manifest["primary"][split] = str(out_path)

        if external_sets:
            external_dir = export_root / "external"
            _safe_mkdir(external_dir)
            for sid, samples in external_sets.items():
                out_path = external_dir / f"{sid}.jsonl"
                if out_path.exists() and not overwrite:
                    self.log.warning("Skip export (exists): %s", str(out_path))
                else:
                    export_jsonl(samples, out_path)
                manifest["external"][sid] = str(out_path)

        return manifest


# -----------------------------
# CLI
# -----------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARTPipeline datasets: load, normalize, split, and export (ACL reproducibility).")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config. If omitted, use defaults.")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotted-path overrides, key=value.")
    p.add_argument("--materialize", action="store_true", help="Materialize runs/{exp_id}/ and write snapshots via config.")
    p.add_argument("--include-external", action="store_true", help="Also load external suites (BOLD/HB/RTP/HateCheck).")
    p.add_argument("--external-only", action="store_true", help="Load only external suites (skip primary).")

    p.add_argument("--export-normalized", action="store_true", help="Export normalized datasets to runs/{exp_id}/datasets/.")
    p.add_argument("--export-dir", type=str, default=None, help="Override export directory (default: runs/{exp_id}/datasets).")
    p.add_argument("--write-cards", action="store_true", help="Write dataset cards to runs/{exp_id}/configs/dataset_cards.json.")
    p.add_argument("--no-file-hash", action="store_true", help="Do not compute file hashes in dataset cards (faster).")

    p.add_argument("--print-summary", action="store_true", help="Print a compact dataset summary to stdout.")
    return p.parse_args()


def _compact_summary(primary_splits: Optional[Dict[str, List[NormalizedSample]]],
                     external_sets: Optional[Dict[str, List[NormalizedSample]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if primary_splits:
        out["primary"] = {k: len(v) for k, v in primary_splits.items()}
        test = primary_splits.get("test", [])
        out["primary_test_label_counts"] = dict(
            _value_counts([s.gold_label for s in test if s.gold_label is not None], max_items=10)
        )
    if external_sets:
        out["external"] = {k: len(v) for k, v in external_sets.items()}
    return out


def main() -> None:
    args = _cli()

    # Load config
    if args.config:
        cfg = DARTPipelineConfig.from_dict(load_config_file(Path(args.config)))
    else:
        cfg = build_default_config()

    cfg = apply_overrides(cfg, args.override)

    # Materialize run directories if requested; otherwise validate quickly.
    if args.materialize:
        cfg.materialize_run(create_dirs=True)
        logs_dir = Path(cfg.derived.get("paths", {}).get("logs_dir", Path(cfg.paths.root()) / "runs" / (cfg.run.exp_id or "run") / "logs"))
        log = configure_logging(logs_dir, level=logging.INFO, console=True)
    else:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger("dartpipeline.datasets.cli")
        cfg.data.ensure_defaults(cfg.paths)
        cfg.validate()

    dm = DatasetManager(cfg, log=log)

    primary_splits: Optional[Dict[str, List[NormalizedSample]]] = None
    external_sets: Optional[Dict[str, List[NormalizedSample]]] = None

    if not args.external_only:
        primary_all = dm.load_primary()
        primary_splits = dm.split_primary(primary_all)

    if args.include_external or args.external_only:
        external_sets = dm.load_all_external()

    if args.print_summary:
        print(json.dumps(_compact_summary(primary_splits, external_sets), ensure_ascii=False, indent=2))

    if args.write_cards:
        if not args.materialize:
            raise RuntimeError("--write-cards requires --materialize to know run directory.")
        cards = dm.build_dataset_cards(primary_splits or {"train": [], "val": [], "test": []}, external_sets, compute_file_hash=not args.no_file_hash)
        cards_path = Path(cfg.derived["paths"]["configs_dir"]) / "dataset_cards.json"
        _json_dump_file({k: v.to_dict() for k, v in cards.items()}, cards_path)
        log.info("Wrote dataset cards: %s", str(cards_path))

    if args.export_normalized:
        if not args.materialize:
            raise RuntimeError("--export-normalized requires --materialize to know run directory.")
        export_root = Path(args.export_dir) if args.export_dir else (Path(cfg.derived["paths"]["run_dir"]) / "datasets")
        manifest = dm.export_normalized(primary_splits or {"train": [], "val": [], "test": []}, external_sets, export_root=export_root, overwrite=True)
        manifest_path = Path(cfg.derived["paths"]["configs_dir"]) / "datasets.export_manifest.json"
        _json_dump_file(manifest, manifest_path)
        log.info("Exported normalized datasets under: %s", str(export_root))
        log.info("Wrote export manifest: %s", str(manifest_path))


if __name__ == "__main__":
    main()