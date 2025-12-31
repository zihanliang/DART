# model/evaluate.py
"""
DARTPipeline Stage D: Evaluation (Benchmark Suite + Reporting)

It supports:
1) Single-model inference outputs (model/inference.py):
   - outputs.jsonl with fields:
       sample_id, policy_id, source, split, prompt, gold_label, meta,
       model_id, raw_text, parsed_analysis, parsed_conclusion, parse_ok, parse_issues, ...

2) Paired inference outputs (model/inference.py --paired):
   - outputs.jsonl with fields:
       sample_id, policy_id, source, split, prompt, gold_label, meta,
       paired: {A:{...}, B:{...}}

Core evaluation products:
- Decision quality on labeled benchmarks (accuracy, macro-F1, balanced accuracy, YES precision/recall/F1)
- Format compliance (parse_ok rate, abstain rate)
- Slices: by benchmark/source, by group (descriptive vs normative vs external), by split, by gold_label
- Paired significance tests vs baseline:
    * McNemar exact test (paired classification)
    * Bootstrap CI for delta accuracy/F1 (optional)
- Exports:
    * CSV tables
    * JSON summary
    * LaTeX table snippets (optional)

Output directory:
  runs/{exp_id}/evaluation/{run_name}/

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

# Optional tqdm
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

logger = logging.getLogger("dartpipeline.evaluate")


# -----------------------------
# Basic helpers
# -----------------------------

Label = Literal["YES", "NO"]
Side = Literal["A", "B"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    _safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def normalize_label(x: Any) -> Optional[Label]:
    if isinstance(x, str):
        u = x.strip().upper()
        if u in ("YES", "NO"):
            return u  # type: ignore[return-value]
        if u in ("DIFF", "EQUAL"):
            return "YES" if u == "DIFF" else "NO"
    return None


def record_key(sample_id: str, policy_id: Optional[str]) -> str:
    pid = policy_id if policy_id is not None else "policy_unknown"
    return f"{sample_id}|{pid}"


def derive_group_from_source(source: Optional[str]) -> str:
    """
    DARTPipeline slice: descriptive vs normative (D* vs N*). External otherwise.
    """
    if not source:
        return "unknown"
    s = str(source).strip()
    if not s:
        return "unknown"
    if s.upper().startswith("D"):
        return "descriptive"
    if s.upper().startswith("N"):
        return "normative"
    return "external"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def fmt_float(x: Any, nd: int = 4) -> Any:
    try:
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return x
        return round(float(x), nd)
    except Exception:
        return x


# -----------------------------
# Prediction structures
# -----------------------------

@dataclass
class Pred:
    sample_id: str
    policy_id: Optional[str]
    source: Optional[str]
    split: Optional[str]
    gold_label: Optional[Label]
    pred_label: Optional[Label]
    parse_ok: bool
    model_id: str
    raw_text: Optional[str] = None
    parsed_analysis: Optional[str] = None
    meta: Any = None

    def key(self) -> str:
        return record_key(self.sample_id, self.policy_id)


@dataclass
class PairedPred:
    """
    A paired prediction aligned on the same sample_id|policy_id.
    """
    sample_id: str
    policy_id: Optional[str]
    source: Optional[str]
    split: Optional[str]
    gold_label: Optional[Label]
    meta: Any
    A: Pred
    B: Pred

    def key(self) -> str:
        return record_key(self.sample_id, self.policy_id)


def _pred_from_single_row(row: Dict[str, Any], forced_model_id: Optional[str] = None) -> Pred:
    sid = str(row.get("sample_id") or "").strip()
    pid = row.get("policy_id")
    source = row.get("source")
    split = row.get("split")
    gold = normalize_label(row.get("gold_label"))
    parse_ok = bool(row.get("parse_ok", False))
    pred_lab = normalize_label(row.get("parsed_conclusion")) if parse_ok else None
    model_id = forced_model_id or str(row.get("model_id") or "model")

    return Pred(
        sample_id=sid,
        policy_id=pid,
        source=str(source) if source is not None else None,
        split=str(split) if split is not None else None,
        gold_label=gold,
        pred_label=pred_lab,
        parse_ok=parse_ok,
        model_id=model_id,
        raw_text=row.get("raw_text"),
        parsed_analysis=row.get("parsed_analysis"),
        meta=row.get("meta"),
    )


def _pred_from_paired_row(row: Dict[str, Any], side: Side, default_model_id: str) -> Pred:
    sid = str(row.get("sample_id") or "").strip()
    pid = row.get("policy_id")
    source = row.get("source")
    split = row.get("split")
    gold = normalize_label(row.get("gold_label"))

    paired = row.get("paired") or {}
    sd = paired.get(side) or {}
    if not isinstance(sd, dict):
        sd = {}

    parse_ok = bool(sd.get("parse_ok", False))
    pred_lab = normalize_label(sd.get("parsed_conclusion")) if parse_ok else None
    model_id = str(sd.get("model_id") or default_model_id)

    return Pred(
        sample_id=sid,
        policy_id=pid,
        source=str(source) if source is not None else None,
        split=str(split) if split is not None else None,
        gold_label=gold,
        pred_label=pred_lab,
        parse_ok=parse_ok,
        model_id=model_id,
        raw_text=sd.get("raw_text"),
        parsed_analysis=sd.get("parsed_analysis"),
        meta=row.get("meta"),
    )


def load_predictions_any(path: Path, *, forced_model_id: Optional[str] = None) -> Tuple[bool, List[Union[Pred, PairedPred]]]:
    """
    Returns:
      is_paired, records
    """
    rows = read_jsonl(path)
    if not rows:
        return False, []

    is_paired = any(isinstance(r.get("paired"), dict) for r in rows)

    if is_paired:
        out: List[Union[Pred, PairedPred]] = []
        for r in rows:
            if "paired" not in r:
                continue
            sid = str(r.get("sample_id") or "").strip()
            if not sid:
                continue
            a = _pred_from_paired_row(r, "A", default_model_id="A")
            b = _pred_from_paired_row(r, "B", default_model_id="B")
            out.append(
                PairedPred(
                    sample_id=sid,
                    policy_id=r.get("policy_id"),
                    source=str(r.get("source")) if r.get("source") is not None else None,
                    split=str(r.get("split")) if r.get("split") is not None else None,
                    gold_label=normalize_label(r.get("gold_label")),
                    meta=r.get("meta"),
                    A=a,
                    B=b,
                )
            )
        return True, out

    # Single
    out2: List[Union[Pred, PairedPred]] = []
    for r in rows:
        sid = str(r.get("sample_id") or "").strip()
        if not sid:
            continue
        out2.append(_pred_from_single_row(r, forced_model_id=forced_model_id))
    return False, out2


def align_two_single_runs(a: List[Pred], b: List[Pred], a_name: str, b_name: str) -> List[PairedPred]:
    idx_a: Dict[str, Pred] = {p.key(): p for p in a}
    idx_b: Dict[str, Pred] = {p.key(): p for p in b}
    keys = sorted(set(idx_a.keys()) & set(idx_b.keys()))

    out: List[PairedPred] = []
    for k in keys:
        pa = idx_a[k]
        pb = idx_b[k]
        # Ensure model_id labels
        pa.model_id = pa.model_id or a_name
        pb.model_id = pb.model_id or b_name
        out.append(
            PairedPred(
                sample_id=pa.sample_id,
                policy_id=pa.policy_id,
                source=pa.source or pb.source,
                split=pa.split or pb.split,
                gold_label=pa.gold_label or pb.gold_label,
                meta=pa.meta if pa.meta is not None else pb.meta,
                A=pa,
                B=pb,
            )
        )
    return out


# -----------------------------
# Metrics: confusion + F1
# -----------------------------

@dataclass
class Confusion:
    tp_yes: int = 0
    fp_yes: int = 0
    fn_yes: int = 0
    tn_yes: int = 0
    abstain: int = 0
    n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def compute_confusion(items: List[Pred]) -> Confusion:
    c = Confusion()
    for p in items:
        c.n += 1
        gold = p.gold_label
        pred = p.pred_label

        if pred is None:
            c.abstain += 1
            continue

        if pred == "YES":
            if gold == "YES":
                c.tp_yes += 1
            elif gold == "NO":
                c.fp_yes += 1
        else:  # pred == NO
            if gold == "YES":
                c.fn_yes += 1
            elif gold == "NO":
                c.tn_yes += 1
    return c


def safe_div0(a: float, b: float) -> float:
    """
    Division with zero_division=0 semantics (sklearn-like):
    if b == 0 -> return 0.0
    """
    try:
        return (a / b) if b != 0 else 0.0
    except Exception:
        return 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # zero_division=0: if denom=0 -> 0
    prec = safe_div0(tp, tp + fp)
    rec = safe_div0(tp, tp + fn)
    f1 = safe_div0(2.0 * prec * rec, prec + rec)
    return prec, rec, f1

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # zero_division=0: if denom=0 -> 0
    prec = safe_div0(tp, tp + fp)
    rec = safe_div0(tp, tp + fn)
    f1 = safe_div0(2.0 * prec * rec, prec + rec)
    return prec, rec, f1

def accuracy_including_abstain(conf: Confusion) -> float:
    """
    Accuracy including abstain cases (abstains count as incorrect).
    (tp + tn) / n
    """
    if conf.n == 0:
        return float("nan")
    return (conf.tp_yes + conf.tn_yes) / conf.n


def accuracy_on_parsed(conf: Confusion) -> float:
    """
    Accuracy on parsed items only (excluding abstain).
    (tp + tn) / (tp + fp + fn + tn)
    """
    denom = conf.tp_yes + conf.fp_yes + conf.fn_yes + conf.tn_yes
    if denom == 0:
        return float("nan")
    return (conf.tp_yes + conf.tn_yes) / denom


def compute_macro_f1_on_parsed(conf: Confusion) -> float:
    """
    Macro-averaged F1 on parsed items.
    Average of F1 for YES and F1 for NO.
    """
    # YES class
    yes_prec, yes_rec, yes_f1 = precision_recall_f1(conf.tp_yes, conf.fp_yes, conf.fn_yes)
    
    # NO class (tn is tp for NO, fn_yes is fp for NO, fp_yes is fn for NO)
    tp_no = conf.tn_yes
    fp_no = conf.fn_yes
    fn_no = conf.fp_yes
    no_prec, no_rec, no_f1 = precision_recall_f1(tp_no, fp_no, fn_no)
    
    # If both F1 scores are valid, return their average
    if yes_f1 == yes_f1 and no_f1 == no_f1:  # Check for NaN
        return (yes_f1 + no_f1) / 2.0
    return float("nan")


def balanced_accuracy_on_parsed(conf: Confusion) -> float:
    """
    Balanced accuracy on parsed items.
    (sensitivity + specificity) / 2 = (recall_yes + recall_no) / 2
    """
    # Sensitivity (recall for YES)
    recall_yes = safe_div0(conf.tp_yes, conf.tp_yes + conf.fn_yes)
    
    # Specificity (recall for NO)
    recall_no = safe_div0(conf.tn_yes, conf.tn_yes + conf.fp_yes)
    
    if recall_yes == recall_yes and recall_no == recall_no:  # Check for NaN
        return (recall_yes + recall_no) / 2.0
    return float("nan")


def prf_for_yes(conf: Confusion) -> Tuple[float, float, float]:
    """
    Precision, recall, F1 for the YES class.
    """
    return precision_recall_f1(conf.tp_yes, conf.fp_yes, conf.fn_yes)

@dataclass
class DecisionReport:
    n: int
    labeled_n: int
    unlabeled_n: int

    parse_ok_rate: float
    abstain_rate: float

    acc_including_abstain: float
    acc_on_parsed: float
    macro_f1_on_parsed: float
    balanced_acc_on_parsed: float

    yes_precision: float
    yes_recall: float
    yes_f1: float
    yes_rate_on_parsed: float

    gold_n_yes: int = 0
    gold_n_no: int = 0
    gold_yes_rate: float = 0.0
    gold_single_class: bool = False

    pred_n_yes: int = 0
    pred_n_no: int = 0
    pred_yes_rate: float = 0.0
    pred_single_class: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def csv_fields(cls) -> List[str]:
        return [
            "acc_on_parsed",
            "macro_f1_on_parsed",
            "abstain_rate",
            "parse_ok_rate",
            "acc_including_abstain",
            "balanced_acc_on_parsed",
            "yes_rate_on_parsed",
            "yes_precision",
            "yes_recall",
            "yes_f1",
            "gold_n_yes",
            "gold_n_no",
            "gold_yes_rate",
            "gold_single_class",
            "pred_n_yes",
            "pred_n_no",
            "pred_yes_rate",
            "pred_single_class",
            "n",
            "labeled_n",
            "unlabeled_n",
        ]


def compute_decision_report(items: List[Pred]) -> DecisionReport:
    n = len(items)
    labeled = [p for p in items if p.gold_label in ("YES", "NO")]
    unlabeled = [p for p in items if p.gold_label not in ("YES", "NO")]

    labeled_n = len(labeled)
    unlabeled_n = len(unlabeled)

    # --- gold distribution on ALL labeled (independent of parse_ok) ---
    gold_n_yes = sum(1 for p in labeled if p.gold_label == "YES")
    gold_n_no = sum(1 for p in labeled if p.gold_label == "NO")
    gold_total = gold_n_yes + gold_n_no
    gold_yes_rate = (gold_n_yes / gold_total) if gold_total > 0 else 0.0
    gold_single_class = (gold_total > 0) and (gold_n_yes == 0 or gold_n_no == 0)

    # --- parsed subset ---
    labeled_parsed = [p for p in labeled if p.parse_ok]
    parsed_n = len(labeled_parsed)
    parse_ok_rate = (parsed_n / labeled_n) if labeled_n > 0 else 0.0

    abstain_n = sum(1 for p in labeled_parsed if p.pred_label is None)
    abstain_rate = (abstain_n / parsed_n) if parsed_n > 0 else 0.0

    # --- prediction distribution on parsed+labeled (excluding abstain) ---
    pred_n_yes = sum(1 for p in labeled_parsed if p.pred_label == "YES")
    pred_n_no = sum(1 for p in labeled_parsed if p.pred_label == "NO")
    pred_total = pred_n_yes + pred_n_no
    pred_yes_rate = (pred_n_yes / pred_total) if pred_total > 0 else 0.0
    pred_single_class = (pred_total > 0) and (pred_n_yes == 0 or pred_n_no == 0)

    # --- existing metric computations (keep your current logic) ---
    conf_all = compute_confusion(labeled)
    conf_parsed = compute_confusion(labeled_parsed)

    acc_including_abstain = accuracy_including_abstain(conf_all)
    acc_on_parsed = accuracy_on_parsed(conf_parsed)
    macro_f1 = compute_macro_f1_on_parsed(conf_parsed)  # Changed function name
    balanced_acc = balanced_accuracy_on_parsed(conf_parsed)  # Changed variable name

    yes_prec, yes_rec, yes_f1 = prf_for_yes(conf_parsed)
    yes_rate_on_parsed = pred_yes_rate

    return DecisionReport(
        n=n,
        labeled_n=labeled_n,
        unlabeled_n=unlabeled_n,
        parse_ok_rate=parse_ok_rate,
        abstain_rate=abstain_rate,
        acc_including_abstain=acc_including_abstain,
        acc_on_parsed=acc_on_parsed,
        macro_f1_on_parsed=macro_f1,  # Use the renamed variable
        balanced_acc_on_parsed=balanced_acc,  # Use the renamed variable
        yes_precision=yes_prec,
        yes_recall=yes_rec,
        yes_f1=yes_f1,
        yes_rate_on_parsed=yes_rate_on_parsed,
        gold_n_yes=gold_n_yes,
        gold_n_no=gold_n_no,
        gold_yes_rate=gold_yes_rate,
        gold_single_class=gold_single_class,
        pred_n_yes=pred_n_yes,
        pred_n_no=pred_n_no,
        pred_yes_rate=pred_yes_rate,
        pred_single_class=pred_single_class,
    )

# -----------------------------
# McNemar exact test (paired significance)
# -----------------------------

@dataclass
class McNemarResult:
    n: int
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    p_two_sided: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """
    CDF of Binomial(n, p) at k (inclusive). Exact summation; adequate for typical dataset sizes.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # Use log-space for stability when n is large.
    # Here n is usually <= tens of thousands; summation is still fine with logs.
    logp = math.log(p)
    logq = math.log(1.0 - p)
    total = 0.0
    for i in range(0, k + 1):
        # log(C(n,i) p^i (1-p)^(n-i))
        lc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        total += math.exp(lc + i * logp + (n - i) * logq)
    return total


def mcnemar_exact(paired: List[PairedPred]) -> McNemarResult:
    """
    Exact two-sided McNemar test based on binomial distribution of discordant pairs.
    Uses labeled items only.
    """
    b = 0
    c = 0
    n = 0

    for e in paired:
        gold = e.gold_label
        if gold is None:
            continue
        n += 1

        a_correct = (e.A.pred_label is not None) and (e.A.pred_label == gold)
        b_correct = (e.B.pred_label is not None) and (e.B.pred_label == gold)

        if a_correct and (not b_correct):
            b += 1
        elif (not a_correct) and b_correct:
            c += 1

    d = b + c
    if d == 0:
        return McNemarResult(n=n, b=b, c=c, p_two_sided=1.0)

    # Two-sided exact p: 2 * min( P(X <= min(b,c)), P(X >= max(b,c)) ) where X~Bin(d,0.5)
    m = min(b, c)
    p_lower = _binom_cdf(m, d, 0.5)
    # symmetry: p_upper = P(X >= max(b,c)) = P(X <= min(b,c)) = p_lower
    p = min(1.0, 2.0 * p_lower)
    return McNemarResult(n=n, b=b, c=c, p_two_sided=p)


# -----------------------------
# Bootstrap CI for deltas
# -----------------------------

@dataclass
class BootstrapCI:
    n_boot: int
    alpha: float
    mean: float
    lo: float
    hi: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def bootstrap_delta(
    paired: List[PairedPred],
    metric_fn,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapCI:
    """
    Bootstrap CI for delta = metric(B) - metric(A) on paired examples.
    """
    rnd = random.Random(seed)
    vals: List[float] = []
    N = len(paired)
    if N == 0:
        return BootstrapCI(n_boot=n_boot, alpha=alpha, mean=float("nan"), lo=float("nan"), hi=float("nan"))

    for _ in range(n_boot):
        sample = [paired[rnd.randrange(N)] for _ in range(N)]
        v = metric_fn(sample)
        vals.append(float(v))

    vals_sorted = sorted(vals)
    mean = sum(vals_sorted) / len(vals_sorted)
    lo_idx = int((alpha / 2.0) * (len(vals_sorted) - 1))
    hi_idx = int((1.0 - alpha / 2.0) * (len(vals_sorted) - 1))
    return BootstrapCI(n_boot=n_boot, alpha=alpha, mean=mean, lo=vals_sorted[lo_idx], hi=vals_sorted[hi_idx])


def _acc_on_parsed_side(items: List[Pred]) -> float:
    labeled = [p for p in items if p.gold_label is not None and p.pred_label is not None]
    if not labeled:
        return float("nan")
    return sum(1 for p in labeled if p.pred_label == p.gold_label) / len(labeled)


def delta_accuracy_on_parsed(paired: List[PairedPred]) -> float:
    a_items = [e.A for e in paired]
    b_items = [e.B for e in paired]
    return _acc_on_parsed_side(b_items) - _acc_on_parsed_side(a_items)


def _macro_f1_on_parsed(items: List[Pred]) -> float:
    labeled = [p for p in items if p.gold_label is not None]
    if not labeled:
        return float("nan")
    conf = compute_confusion(labeled)
    # YES
    yes_prec, yes_rec, yes_f1 = precision_recall_f1(conf.tp_yes, conf.fp_yes, conf.fn_yes)
    # NO
    tp_no = conf.tn_yes
    fp_no = conf.fn_yes
    fn_no = conf.fp_yes
    no_prec, no_rec, no_f1 = precision_recall_f1(tp_no, fp_no, fn_no)
    if yes_f1 != yes_f1 or no_f1 != no_f1:
        return float("nan")
    return (yes_f1 + no_f1) / 2.0


def delta_macro_f1_on_parsed(paired: List[PairedPred]) -> float:
    return _macro_f1_on_parsed([e.B for e in paired]) - _macro_f1_on_parsed([e.A for e in paired])


# -----------------------------
# Slicing / reporting
# -----------------------------

def slice_key_funcs() -> Dict[str, Any]:
    return {
        "all": lambda p: "all",
        "policy_id": lambda p: str(p.policy_id) if p.policy_id is not None else "policy_unknown",
        "source": lambda p: p.source if p.source is not None else "unknown",
        "group": lambda p: derive_group_from_source(p.source),
        "split": lambda p: p.split if p.split is not None else "unknown",
        "gold_label": lambda p: p.gold_label if p.gold_label is not None else "unknown",
    }


def group_preds(items: List[Pred]) -> Dict[Tuple[str, str], List[Pred]]:
    """
    Returns dict keyed by (slice_name, slice_value).
    """
    out: Dict[Tuple[str, str], List[Pred]] = {}
    fns = slice_key_funcs()
    for sname, fn in fns.items():
        for p in items:
            sval = str(fn(p))
            out.setdefault((sname, sval), []).append(p)
    return out


def group_pairs(paired: List[PairedPred]) -> Dict[Tuple[str, str], List[PairedPred]]:
    """
    Group paired examples using the same slice functions (on meta fields).
    """
    out: Dict[Tuple[str, str], List[PairedPred]] = {}
    for sname, fn in slice_key_funcs().items():
        for e in paired:
            # Apply on a Pred-like view (use A for fields)
            ref = e.A
            sval = str(fn(ref))
            out.setdefault((sname, sval), []).append(e)
    return out


# -----------------------------
# Audit summary ingestion
# -----------------------------

def read_audit_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_latest_audit_summary(run_dir: Path, audit_run_name: Optional[str]) -> Optional[Path]:
    """
    If audit_run_name is provided, try runs/{exp_id}/audit/{name}/summary.json
    Else attempt to locate the most recently modified summary.json under audit/.
    """
    audit_root = run_dir / "audit"
    if not audit_root.exists():
        return None
    if audit_run_name:
        p = audit_root / audit_run_name / "summary.json"
        return p if p.exists() else None

    # Find newest summary.json
    candidates = list(audit_root.glob("**/summary.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# Main evaluation
# -----------------------------

@dataclass
class EvalConfig:
    min_slice_n: int = 30
    export_latex: bool = True
    bootstrap: bool = True
    bootstrap_n: int = 2000
    bootstrap_seed: int = 42
    bootstrap_alpha: float = 0.05


def build_run_dir(cfg: DARTPipelineConfig) -> Path:
    if isinstance(getattr(cfg, "derived", None), dict):
        p = cfg.derived.get("paths", {}).get("run_dir")
        if p:
            return Path(p)
    root = Path(cfg.paths.root())
    exp_id = getattr(cfg.run, "exp_id", None) or "run"
    return root / "runs" / exp_id


def default_eval_out_dir(cfg: DARTPipelineConfig, run_name: str) -> Path:
    return build_run_dir(cfg) / "evaluation" / run_name


def filter_preds(
    items: List[Pred],
    *,
    policy_filter: Optional[str],
    splits: Optional[List[str]],
    sources: Optional[List[str]],
    max_samples: Optional[int],
) -> List[Pred]:
    out = items

    if policy_filter and policy_filter != "all":
        out = [p for p in out if (str(p.policy_id) if p.policy_id is not None else "policy_unknown") == policy_filter]

    if splits:
        sset = set(splits)
        out = [p for p in out if (p.split or "unknown") in sset]

    if sources:
        srcset = set(sources)
        out = [p for p in out if (p.source or "unknown") in srcset]

    out.sort(key=lambda p: p.key())
    if max_samples is not None and len(out) > max_samples:
        out = out[:max_samples]
    return out


def filter_pairs(
    paired: List[PairedPred],
    *,
    policy_filter: Optional[str],
    splits: Optional[List[str]],
    sources: Optional[List[str]],
    max_samples: Optional[int],
) -> List[PairedPred]:
    out = paired

    if policy_filter and policy_filter != "all":
        out = [e for e in out if (str(e.policy_id) if e.policy_id is not None else "policy_unknown") == policy_filter]

    if splits:
        sset = set(splits)
        out = [e for e in out if (e.split or "unknown") in sset]

    if sources:
        srcset = set(sources)
        out = [e for e in out if (e.source or "unknown") in srcset]

    out.sort(key=lambda e: e.key())
    if max_samples is not None and len(out) > max_samples:
        out = out[:max_samples]
    return out


def render_latex_main_table(
    rows: List[Dict[str, Any]],
    *,
    a_name: Optional[str],
    b_name: Optional[str],
    caption: str,
    label: str,
) -> str:
    """
    Produce a simple LaTeX table snippet for core benchmarks (by source).
    Expected row keys:
      source, model, acc_on_parsed, macro_f1_on_parsed, abstain_rate, parse_ok_rate
      (optional) delta_acc, mcnemar_p, stars
    """
    # Determine ordering: labeled benchmarks first, then external
    def _is_external(src: str) -> int:
        return 1 if derive_group_from_source(src) == "external" else 0

    sources = sorted({r["source"] for r in rows if r.get("source")}, key=lambda s: (_is_external(s), s))
    models = sorted({r["model"] for r in rows if r.get("model")})

    # Build matrix: source x model
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {(r["source"], r["model"]): r for r in rows if r.get("source") and r.get("model")}

    # Table header
    header_cols = ["Benchmark"]
    for m in models:
        header_cols.append(f"{m} Acc")
        header_cols.append(f"{m} F1")
        header_cols.append(f"{m} Abst.")
    col_spec = "l" + "ccc" * len(models)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")

    for src in sources:
        row_cells = [src]
        for m in models:
            r = cells.get((src, m), {})
            acc = r.get("acc_on_parsed")
            f1 = r.get("macro_f1_on_parsed")
            abst = r.get("abstain_rate")
            row_cells.append(f"{acc:.3f}" if isinstance(acc, (int, float)) and acc == acc else "--")
            row_cells.append(f"{f1:.3f}" if isinstance(f1, (int, float)) and f1 == f1 else "--")
            row_cells.append(f"{abst:.3f}" if isinstance(abst, (int, float)) and abst == abst else "--")
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def significance_stars(p: float) -> str:
    if p is None or not isinstance(p, (int, float)) or p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def evaluate_single_run(
    preds: List[Pred],
    *,
    model_name: str,
    min_slice_n: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      overall_json, rows_for_csv (by slice)
    """
    overall = compute_decision_report(preds)
    grouped = group_preds(preds)

    slice_rows: List[Dict[str, Any]] = []
    for (sname, sval), items in grouped.items():
        rep = compute_decision_report(items)
        slice_rows.append({
            "model": model_name,
            "slice": sname,
            "slice_value": sval,
            **{k: rep.to_dict().get(k) for k in [
                "n", "labeled_n", "unlabeled_n",
                "parse_ok_rate", "abstain_rate",
                "acc_including_abstain", "acc_on_parsed",
                "macro_f1_on_parsed", "balanced_acc_on_parsed",
                "yes_precision", "yes_recall", "yes_f1",
                "yes_rate_on_parsed",
            ]},
        })

    overall_json = {
        "model": model_name,
        "overall": overall.to_dict(),
    }
    return overall_json, slice_rows


def evaluate_paired_runs(
    paired: List[PairedPred],
    *,
    a_name: str,
    b_name: str,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - reports for A and B overall
      - McNemar overall
      - optional bootstrap CIs
      - slice-level paired tests where feasible
    """
    a_preds = [e.A for e in paired]
    b_preds = [e.B for e in paired]

    rep_a = compute_decision_report(a_preds)
    rep_b = compute_decision_report(b_preds)
    mcn = mcnemar_exact(paired)

    out: Dict[str, Any] = {
        "A": {"name": a_name, "overall": rep_a.to_dict()},
        "B": {"name": b_name, "overall": rep_b.to_dict()},
        "paired": {
            "n_pairs": len(paired),
            "mcnemar": mcn.to_dict(),
            "delta_acc_on_parsed": rep_b.acc_on_parsed - rep_a.acc_on_parsed,
            "delta_macro_f1_on_parsed": rep_b.macro_f1_on_parsed - rep_a.macro_f1_on_parsed,
        }
    }

    if eval_cfg.bootstrap:
        out["paired"]["bootstrap"] = {
            "delta_acc_on_parsed": bootstrap_delta(
                paired,
                metric_fn=delta_accuracy_on_parsed,
                n_boot=eval_cfg.bootstrap_n,
                seed=eval_cfg.bootstrap_seed,
                alpha=eval_cfg.bootstrap_alpha,
            ).to_dict(),
            "delta_macro_f1_on_parsed": bootstrap_delta(
                paired,
                metric_fn=delta_macro_f1_on_parsed,
                n_boot=eval_cfg.bootstrap_n,
                seed=eval_cfg.bootstrap_seed + 7,
                alpha=eval_cfg.bootstrap_alpha,
            ).to_dict(),
        }

    # Slice-level paired tests
    slices = group_pairs(paired)
    slice_tests: List[Dict[str, Any]] = []
    for (sname, sval), items in slices.items():
        if len(items) < eval_cfg.min_slice_n:
            continue
        m = mcnemar_exact(items)
        # Compute delta on parsed accuracy for slice
        da = delta_accuracy_on_parsed(items)
        df = delta_macro_f1_on_parsed(items)
        row = {
            "slice": sname,
            "slice_value": sval,
            "n": len(items),
            "delta_acc_on_parsed": da,
            "delta_macro_f1_on_parsed": df,
            "mcnemar_p": m.p_two_sided,
            "stars": significance_stars(m.p_two_sided),
            "b": m.b,
            "c": m.c,
        }
        slice_tests.append(row)

    out["paired"]["slice_tests"] = slice_tests
    return out


# -----------------------------
# Prediction discovery
# -----------------------------

def discover_outputs_jsonl(pred_root: Path) -> List[Path]:
    """
    Discover outputs.jsonl under a predictions directory.
    """
    if pred_root.is_file():
        return [pred_root]
    if not pred_root.exists():
        return []
    return sorted(pred_root.glob("**/outputs.jsonl"))


# -----------------------------
# CLI
# -----------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARTPipeline Evaluation: benchmark decision metrics + paired significance tests.")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config. If omitted, defaults are used.")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotted overrides key=value.")
    p.add_argument("--materialize", action="store_true", help="Materialize runs/{exp_id}/ via config.materialize_run().")

    # Inputs
    p.add_argument("--pred", type=str, action="append", default=[],
                   help="Path to an outputs.jsonl (single or paired). Can be provided multiple times.")
    p.add_argument("--pred-dir", type=str, default=None,
                   help="Directory to search for outputs.jsonl under (e.g., runs/<exp_id>/predictions).")

    # Optional explicit two-run compare for significance tests
    p.add_argument("--a-pred", type=str, default=None, help="Single-model outputs.jsonl for baseline A.")
    p.add_argument("--b-pred", type=str, default=None, help="Single-model outputs.jsonl for system B.")
    p.add_argument("--a-name", type=str, default="M0", help="Display name for model A.")
    p.add_argument("--b-name", type=str, default="DART", help="Display name for model B.")
    p.add_argument("--force-model-id-a", type=str, default=None, help="Override model_id for A when loading.")
    p.add_argument("--force-model-id-b", type=str, default=None, help="Override model_id for B when loading.")

    # Filters
    p.add_argument("--policy-filter", type=str, default="all",
                   help="Filter to a specific policy_id (e.g., policy_off/policy_on). Use 'all' for no filter.")
    p.add_argument("--splits", type=str, default=None, help="Comma-separated split filter (train,val,test).")
    p.add_argument("--sources", type=str, default=None, help="Comma-separated source filter (D1,N3,HolisticBias,...).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap sample count deterministically after filtering.")

    # Output
    p.add_argument("--run-name", type=str, required=True, help="Evaluation run name under runs/{exp_id}/evaluation/<run-name>/.")

    # Optional audit ingestion
    p.add_argument("--audit-summary", type=str, default=None,
                   help="Path to an audit summary.json to include in the consolidated report.")
    p.add_argument("--audit-run-name", type=str, default=None,
                   help="If provided, read runs/{exp_id}/audit/<name>/summary.json automatically.")

    # Reporting knobs
    p.add_argument("--min-slice-n", type=int, default=30, help="Minimum examples required to report a slice.")
    p.add_argument("--no-latex", action="store_true", help="Disable LaTeX export.")
    p.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap CI computation.")
    p.add_argument("--bootstrap-n", type=int, default=2000, help="Number of bootstrap replicates.")
    p.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap seed.")
    p.add_argument("--bootstrap-alpha", type=float, default=0.05, help="Alpha for CI (0.05 => 95% CI).")

    return p.parse_args()


def main() -> None:
    args = _cli()

    # -------------------------
    # Load config
    # -------------------------
    if args.config:
        cfg = DARTPipelineConfig.from_dict(load_config_file(Path(args.config)))
    else:
        cfg = build_default_config()
    cfg = apply_overrides(cfg, args.override)

    # -------------------------
    # Materialize / validate
    # -------------------------
    if args.materialize:
        cfg.materialize_run(create_dirs=True)
    else:
        if hasattr(cfg, "data") and hasattr(cfg.data, "ensure_defaults"):
            cfg.data.ensure_defaults(cfg.paths)
        if hasattr(cfg, "validate"):
            cfg.validate()

    run_dir = build_run_dir(cfg)
    logs_dir = run_dir / "logs"
    configure_logging(logs_dir, level=logging.INFO, console=True)
    set_global_seed(
        int(getattr(cfg.run, "seed", 42)),
        deterministic=bool(getattr(cfg.run, "deterministic", True)),
    )

    out_dir = default_eval_out_dir(cfg, args.run_name)
    _safe_mkdir(out_dir)

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    eval_cfg = EvalConfig(
        min_slice_n=int(args.min_slice_n),
        export_latex=not args.no_latex,
        bootstrap=not args.no_bootstrap,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_alpha=float(args.bootstrap_alpha),
    )

    # -------------------------
    # Collect prediction paths
    # -------------------------
    pred_paths: List[Path] = []
    for p in args.pred:
        pred_paths.append(Path(p).expanduser().resolve())

    if args.pred_dir:
        pred_paths += discover_outputs_jsonl(Path(args.pred_dir).expanduser().resolve())

    # If explicit compare is provided, we do paired evaluation even if pred_paths is empty.
    explicit_compare = bool(args.a_pred and args.b_pred)

    if not pred_paths and not explicit_compare:
        # Reasonable default: search under runs/{exp_id}/predictions
        pred_root_guess = run_dir / "predictions"
        pred_paths = discover_outputs_jsonl(pred_root_guess)

    if not pred_paths and not explicit_compare:
        raise RuntimeError("No prediction outputs found. Provide --pred / --pred-dir, or --a-pred and --b-pred.")

    # -------------------------
    # Optional audit summary ingestion
    # -------------------------
    audit_summary_obj = None
    if args.audit_summary:
        audit_summary_obj = read_audit_summary(Path(args.audit_summary).expanduser().resolve())
    else:
        p = discover_latest_audit_summary(run_dir, args.audit_run_name)
        if p is not None:
            audit_summary_obj = read_audit_summary(p)

    # -------------------------
    # Helpers
    # -------------------------
    def _decision_row(run_tag: str, source: str, model: str, rep: "DecisionReport") -> Dict[str, Any]:
        row: Dict[str, Any] = {"run_tag": run_tag, "source": source, "model": model}
        # Preferred: DecisionReport.to_dict()
        if hasattr(rep, "to_dict") and callable(getattr(rep, "to_dict")):
            row.update(rep.to_dict())
        else:
            # Fallback: dataclasses.asdict
            row.update(dataclasses.asdict(rep))
        return row

    def _main_table_fieldnames() -> List[str]:
        base = ["run_tag", "source", "model"]
        if hasattr(DecisionReport, "csv_fields") and callable(getattr(DecisionReport, "csv_fields")):
            return base + list(DecisionReport.csv_fields())
        # Fallback (older schema)
        return base + [
            "acc_on_parsed",
            "macro_f1_on_parsed",
            "abstain_rate",
            "parse_ok_rate",
            "acc_including_abstain",
            "balanced_acc_on_parsed",
            "yes_rate_on_parsed",
            "yes_precision",
            "yes_recall",
            "yes_f1",
            "gold_n_yes",
            "gold_n_no",
            "gold_yes_rate",
            "gold_single_class",
        ]

    # -------------------------
    # Evaluate each prediction file
    # -------------------------
    single_run_summaries: List[Dict[str, Any]] = []
    slice_rows_all: List[Dict[str, Any]] = []
    main_table_rows: List[Dict[str, Any]] = []

    for path in pred_paths:
        is_paired, recs = load_predictions_any(path)
        run_tag = path.parent.name  # typical: predictions/<run_name>/outputs.jsonl
        logger.info("Loaded: %s | paired=%s | n=%d", str(path), is_paired, len(recs))

        if is_paired:
            paired = [r for r in recs if isinstance(r, PairedPred)]
            paired = filter_pairs(
                paired,
                policy_filter=args.policy_filter,
                splits=splits,
                sources=sources,
                max_samples=args.max_samples,
            )

            paired_report = evaluate_paired_runs(
                paired,
                a_name=args.a_name,
                b_name=args.b_name,
                eval_cfg=eval_cfg,
            )
            single_run_summaries.append({
                "type": "paired",
                "run_tag": run_tag,
                "path": str(path),
                "report": paired_report,
            })

            # Per-source rows for BOTH sides (A and B), using the same schema
            by_source_a: Dict[str, List[Pred]] = {}
            by_source_b: Dict[str, List[Pred]] = {}
            for e in paired:
                src = e.source or "unknown"
                by_source_a.setdefault(src, []).append(e.A)
                by_source_b.setdefault(src, []).append(e.B)

            for src, items in by_source_a.items():
                rep = compute_decision_report(items)
                main_table_rows.append(_decision_row(run_tag, src, args.a_name, rep))

            for src, items in by_source_b.items():
                rep = compute_decision_report(items)
                main_table_rows.append(_decision_row(run_tag, src, args.b_name, rep))

        else:
            preds = [r for r in recs if isinstance(r, Pred)]
            preds = filter_preds(
                preds,
                policy_filter=args.policy_filter,
                splits=splits,
                sources=sources,
                max_samples=args.max_samples,
            )

            # Determine model name (prefer the dominant model_id)
            model_ids = [p.model_id for p in preds if p.model_id]
            model_name = model_ids[0] if model_ids else run_tag

            overall_json, slice_rows = evaluate_single_run(preds, model_name=model_name, min_slice_n=eval_cfg.min_slice_n)
            overall_json["type"] = "single"
            overall_json["run_tag"] = run_tag
            overall_json["path"] = str(path)
            single_run_summaries.append(overall_json)
            slice_rows_all.extend(slice_rows)

            # Main table rows by source
            by_source: Dict[str, List[Pred]] = {}
            for p in preds:
                src = p.source or "unknown"
                by_source.setdefault(src, []).append(p)

            for src, items in by_source.items():
                rep = compute_decision_report(items)
                main_table_rows.append(_decision_row(run_tag, src, model_name, rep))

    # -------------------------
    # Explicit compare evaluation (A vs B) if provided
    # -------------------------
    compare_report = None
    compare_rows: List[Dict[str, Any]] = []
    if explicit_compare:
        a_path = Path(args.a_pred).expanduser().resolve()
        b_path = Path(args.b_pred).expanduser().resolve()

        is_paired_a, rec_a = load_predictions_any(a_path, forced_model_id=args.force_model_id_a)
        is_paired_b, rec_b = load_predictions_any(b_path, forced_model_id=args.force_model_id_b)
        if is_paired_a or is_paired_b:
            raise ValueError("Explicit compare (--a-pred/--b-pred) expects single-model outputs.jsonl for both runs.")

        a_preds = filter_preds(
            [r for r in rec_a if isinstance(r, Pred)],
            policy_filter=args.policy_filter,
            splits=splits,
            sources=sources,
            max_samples=args.max_samples,
        )
        b_preds = filter_preds(
            [r for r in rec_b if isinstance(r, Pred)],
            policy_filter=args.policy_filter,
            splits=splits,
            sources=sources,
            max_samples=args.max_samples,
        )

        paired_cmp = align_two_single_runs(a_preds, b_preds, a_name=args.a_name, b_name=args.b_name)
        paired_cmp = filter_pairs(
            paired_cmp,
            policy_filter=args.policy_filter,
            splits=splits,
            sources=sources,
            max_samples=args.max_samples,
        )

        compare_report = evaluate_paired_runs(paired_cmp, a_name=args.a_name, b_name=args.b_name, eval_cfg=eval_cfg)

        by_source_pairs: Dict[str, List[PairedPred]] = {}
        for e in paired_cmp:
            src = e.source or "unknown"
            by_source_pairs.setdefault(src, []).append(e)

        for src, items in by_source_pairs.items():
            if len(items) < eval_cfg.min_slice_n:
                continue
            m = mcnemar_exact(items)
            da = delta_accuracy_on_parsed(items)
            df = delta_macro_f1_on_parsed(items)
            compare_rows.append({
                "source": src,
                "n": len(items),
                "delta_acc_on_parsed": da,
                "delta_macro_f1_on_parsed": df,
                "mcnemar_p": m.p_two_sided,
                "stars": significance_stars(m.p_two_sided),
                "b": m.b,
                "c": m.c,
            })

    # -------------------------
    # Write outputs
    # -------------------------
    summary = {
        "timestamp_utc": utc_now_iso(),
        "policy_filter": args.policy_filter,
        "filters": {"splits": splits, "sources": sources, "max_samples": args.max_samples},
        "eval_cfg": dataclasses.asdict(eval_cfg),
        "runs": single_run_summaries,
        "explicit_compare": compare_report,
        "audit_summary": audit_summary_obj,
    }
    _json_dump_file(summary, out_dir / "summary.json")

    # Flatten slice rows to CSV
    if slice_rows_all:
        fieldnames = [
            "model", "slice", "slice_value",
            "n", "labeled_n", "unlabeled_n",
            "parse_ok_rate", "abstain_rate",
            "acc_including_abstain", "acc_on_parsed",
            "macro_f1_on_parsed", "balanced_acc_on_parsed",
            "yes_precision", "yes_recall", "yes_f1",
            "yes_rate_on_parsed",
        ]
        for r in slice_rows_all:
            for k in fieldnames:
                if isinstance(r.get(k), float):
                    r[k] = fmt_float(r.get(k), 6)
        write_csv(out_dir / "metrics_by_slice.csv", slice_rows_all, fieldnames=fieldnames)

    # Main table rows (by source)
    if main_table_rows:
        fieldnames = _main_table_fieldnames()
        for r in main_table_rows:
            for k in fieldnames:
                if isinstance(r.get(k), float):
                    r[k] = fmt_float(r.get(k), 6)
        write_csv(out_dir / "metrics_by_source.csv", main_table_rows, fieldnames=fieldnames)

    # Compare rows (per-source significance)
    if compare_rows:
        fieldnames = ["source", "n", "delta_acc_on_parsed", "delta_macro_f1_on_parsed", "mcnemar_p", "stars", "b", "c"]
        for r in compare_rows:
            for k in fieldnames:
                if isinstance(r.get(k), float):
                    r[k] = fmt_float(r.get(k), 8)
        write_csv(out_dir / "paired_significance_by_source.csv", compare_rows, fieldnames=fieldnames)

    # Export LaTeX (optional)
    if eval_cfg.export_latex and main_table_rows:
        tables_dir = out_dir / "tables"
        _safe_mkdir(tables_dir)
        run_tags = sorted({r["run_tag"] for r in main_table_rows})

        for tag in run_tags:
            rows = [r for r in main_table_rows if r["run_tag"] == tag]
            models_in_tag = sorted({r.get("model") for r in rows if r.get("model")})

            # If we have both A and B models in this run_tag, pass names so renderer can build a two-model table
            a_name_for_tex = args.a_name if (args.a_name in models_in_tag) else None
            b_name_for_tex = args.b_name if (args.b_name in models_in_tag) else None

            tex = render_latex_main_table(
                rows,
                a_name=a_name_for_tex,
                b_name=b_name_for_tex,
                caption=f"Decision quality by benchmark for {tag}.",
                label=f"tab:decision_{tag}".replace("_", "-"),
            )
            (tables_dir / f"decision_table_{tag}.tex").write_text(tex, encoding="utf-8")

    # Snapshot
    snapshot = {
        "timestamp_utc": utc_now_iso(),
        "out_dir": str(out_dir),
        "pred_paths": [str(p) for p in pred_paths],
        "explicit_compare": {
            "enabled": explicit_compare,
            "a_pred": args.a_pred,
            "b_pred": args.b_pred,
            "a_name": args.a_name,
            "b_name": args.b_name,
        },
        "audit_ingestion": {
            "audit_summary": args.audit_summary,
            "audit_run_name": args.audit_run_name,
            "included": audit_summary_obj is not None,
        },
        "env": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "config_run": dataclasses.asdict(cfg.run),
    }
    _json_dump_file(snapshot, out_dir / "job.snapshot.json")

    logger.info("Evaluation complete. Outputs written under: %s", str(out_dir))

if __name__ == "__main__":
    main()