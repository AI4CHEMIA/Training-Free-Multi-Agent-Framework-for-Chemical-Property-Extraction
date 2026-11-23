"""
evaluation/metrics.py — OmniPropAgent Evaluation Framework v2

Design principles
-----------------
1. The GT is incomplete by construction: a document may contain real property
   values the annotators did not label. Predictions that go BEYOND the GT are
   not automatically wrong — they are "unverified", not "hallucinated".
   Only claims that contradict the source text are penalised as hallucinations.

2. Two evaluation levels:
     L1 — Molecule level  (per predicted / GT (mol, property) pair)
     L2 — Document level  (aggregate of molecule-level results)

3. L1 sub-metrics (for every matched mol+property pair):
     a. Property accuracy  — correct property type identified
     b. Value accuracy     — numerical value within tolerance
     c. Unit accuracy      — unit in correct equivalence family

4. Three additional metrics:
     M4. Extraction completeness   — fraction of GT rows found (no-penalty recall)
     M5. Tuple integrity score     — fraction of matched tuples where ALL four
                                     fields (name, property, value, unit) are
                                     simultaneously correct
     M6. Novel extraction rate     — fraction of predictions that are NOT in GT
                                     (measures discovery potential, not error)

5. Document-level hallucination:
     - Every prediction not matched to GT is flagged as "unverified".
     - An LLM-as-judge call verifies whether the claim is actually supported
       by the source text ("grounded") or invented ("hallucinated").
     - Only LLM-confirmed hallucinations count against the system.
"""

import json
import logging
import re
from collections import defaultdict
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fix_mojibake(s: str) -> str:
    """Recover UTF-8 text that was mis-decoded as Latin-1 (e.g. 'Â°C' → '°C')."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s

_SUP_TABLE = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

def _nv(v: str) -> str:
    v = _fix_mojibake(str(v))
    v = re.sub(r"[\xa0  ​\xad]", " ", v)  # normalise Unicode spaces
    v = re.sub(r"\s+", " ", v.lower()).strip()
    v = v.replace("−", "-").replace("–", "-")
    v = v.replace("∼", "~").replace("≈", "~")
    # Normalize Unicode superscripts to ASCII (for sci notation)
    v = v.translate(_SUP_TABLE)
    # Normalize multiplication signs for sci notation
    v = v.replace("×", "*").replace("·", "*")
    # Convert "A * 10^B" → float string
    m = re.match(
        r"^\s*([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*\*\s*10\^?\s*([+-]?\d+)\s*$",
        v,
    )
    if m:
        try:
            mantissa = float(m.group(1).replace(",", "."))
            result = mantissa * (10 ** int(m.group(2)))
            v = f"{int(result)}" if result == int(result) and abs(result) < 1e15 else f"{result:.10g}"
        except Exception:
            pass
    # Remove all spaces before comma/period processing
    v = re.sub(r"\s+", "", v)
    # Strip thousands separators: "10,800" → "10800", "1,234,567" → "1234567"
    v = re.sub(r",(?=\d{3}(?:[,\d]|$))", "", v)
    # Remaining comma → decimal point (European notation: "2,74" → "2.74")
    v = v.replace(",", ".")
    return v

def _nu(u: str) -> str:
    u = _fix_mojibake(str(u))
    u = u.replace("℃", "°c").replace("℉", "°f")
    # Normalize minus/dash variants to ASCII hyphen (biggest source of unit mismatches)
    u = u.replace("−", "-").replace("–", "-").replace("—", "-")
    # Normalize Unicode superscripts
    u = u.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+"))
    # Normalize vulgar fraction ½ → 0.5
    u = u.replace("½", "0.5")
    # Normalize multiplication/separator dots to middle dot, then to space for comparison
    # U+0387 GREEK ANO TELEIA is also used as a unit separator in some databases
    u = u.replace("·", "·").replace("⋅", "·").replace("∙", "·").replace("·", " ")
    # Strip power caret before digits, parens, or minus (exponent notation in units)
    # "mpa^0.5" → "mpa0.5", "s^-1" → "s-1", "m^-1" → "m-1"
    u = re.sub(r"\^(?=[-\d(])", "", u)
    # Normalize ASCII period as unit separator ("m-1.cm-1" → "m-1 cm-1") but not decimal points
    u = re.sub(r"(?<=\d)\.(?=[a-zA-Zμ])", " ", u)
    return re.sub(r"\s+", " ", u.strip().lower())

def _np(p: str) -> str:
    p = str(p).strip().lower()
    # Remove apostrophes so "young's modulus" == "youngs modulus"
    p = re.sub(r"[''']", "", p)
    # Normalize hyphens to spaces so "auto-ignition" == "auto ignition"
    p = p.replace("-", " ")
    return re.sub(r"\s+", " ", p).strip()


def units_match(u1: str, u2: str) -> bool:
    u1n, u2n = _nu(u1), _nu(u2)
    if u1n == u2n:
        return True
    for cls in config.UNIT_EQUIVALENCES:
        if u1n in cls and u2n in cls:
            return True
    return False


def property_match(p1: str, p2: str) -> bool:
    p1n, p2n = _np(p1), _np(p2)
    if p1n == p2n:
        return True
    for canonical, aliases in config.PROPERTY_SYNONYMS.items():
        # Normalize synonyms with the same _np so hyphen/apostrophe variants match
        grp = {_np(canonical)} | {_np(a) for a in aliases}
        if p1n in grp and p2n in grp:
            return True
    return p1n in p2n or p2n in p1n


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def name_match(n1: str, n2: str,
               n1_long: str = "", n2_long: str = "") -> bool:
    """
    Match chemical names.

    Short-form match is exact only — substring is intentionally excluded to
    prevent false positives like ethanol/methanol or acetic acid/phenylacetic acid.
    Long-name cross-matching handles formula ↔ systematic pairs (MgF2 ↔ magnesium fluoride).
    """
    s1, s2 = _norm_name(n1), _norm_name(n2)
    if s1 == s2:
        return True
    # Cross-match short name against long name (formula vs systematic)
    l1 = _norm_name(n1_long) if n1_long else ""
    l2 = _norm_name(n2_long) if n2_long else ""
    if l1 and (l1 == s2 or (len(l1) > 3 and (l1 in s2 or s2 in l1))):
        return True
    if l2 and (l2 == s1 or (len(l2) > 3 and (l2 in s1 or s1 in l2))):
        return True
    if l1 and l2 and (l1 == l2 or (len(l1) > 3 and (l1 in l2 or l2 in l1))):
        return True
    return False


# Qualifier phrases stripped before numeric parsing — applied to the raw string
# (before _nv removes whitespace) so multi-word phrases like "in the range of" work.
_QUAL_RE = re.compile(
    r"(?i)\b("
    r"approximately|no\s+more\s+than|no\s+less\s+than|"
    r"in\s+the\s+range\s+of|of\s+the\s+order\s+of|"
    r"close\s+enough\s+to|greater\s+than|less\s+than|"
    r"higher\s+than|lower\s+than|more\s+than|"
    r"in\s+the\s+range|close\s+to|"
    r"at\s+least|at\s+most|up\s+to|"
    r"about|approx\.|approx|circa|roughly|around|nearly|exceeds?|ca\."
    r")\s*"
)


_RANGE_RE = re.compile(
    r'^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)-(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$'
)
# Matches "A-Be^N" where A has no exponent but B does (shared-exponent notation).
# e.g. "154-303e6" → both should be ×10^6 → normalize to "154e6-303e6"
_SHARED_EXP_RE = re.compile(
    r'^([+-]?\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)([eE][+-]?\d+)$'
)

def _is_range(v: str) -> bool:
    return bool(_RANGE_RE.match(v))

def _normalize_shared_exp(v: str) -> str:
    """'154-303e6' → '154e6-303e6' when only the second endpoint carries the exponent."""
    m = _SHARED_EXP_RE.match(v)
    if m and not re.search(r'[eE]', m.group(1)):
        return f"{m.group(1)}{m.group(3)}-{m.group(2)}{m.group(3)}"
    return v


def value_match(v1: str, v2: str) -> bool:
    """
    Match values after normalisation.
    - Ranges: normalize shared-exponent form ("154-303e6" → "154e6-303e6"),
      then compare endpoints as floats so "154e6-303e6" == "1.54e8-3.03e8".
    - Scalars: string match first; if that fails, numeric float comparison
      so that "2" == "2.0", "1e8" == "100000000", "5.96e7" == "59600000".
    """
    v1c = _QUAL_RE.sub("", _fix_mojibake(str(v1))).strip()
    v2c = _QUAL_RE.sub("", _fix_mojibake(str(v2))).strip()
    n1, n2 = _nv(v1c), _nv(v2c)
    if n1 == n2:
        return True
    # Normalize shared-exponent range notation before comparison
    n1 = _normalize_shared_exp(n1)
    n2 = _normalize_shared_exp(n2)
    if n1 == n2:
        return True
    # Range vs Range: compare endpoints as floats
    if _is_range(n1) and _is_range(n2):
        m1, m2 = _RANGE_RE.match(n1), _RANGE_RE.match(n2)
        if m1 and m2:
            try:
                return (float(m1.group(1)) == float(m2.group(1)) and
                        float(m1.group(2)) == float(m2.group(2)))
            except (ValueError, TypeError):
                pass
        return False
    # Scalar in Range: if one is a range and the other is a scalar,
    # accept if the scalar falls within the range (paper may report mean, model reports range)
    if _is_range(n1) and not _is_range(n2):
        m = _RANGE_RE.match(n1)
        if m:
            try:
                lo, hi = sorted([float(m.group(1)), float(m.group(2))])
                return lo <= float(n2) <= hi
            except (ValueError, TypeError):
                pass
    if _is_range(n2) and not _is_range(n1):
        m = _RANGE_RE.match(n2)
        if m:
            try:
                lo, hi = sorted([float(m.group(1)), float(m.group(2))])
                return lo <= float(n1) <= hi
            except (ValueError, TypeError):
                pass
    # Scalar numeric comparison (handles trailing .0, e-notation variants)
    if not _is_range(n1) and not _is_range(n2):
        try:
            return float(n1) == float(n2)
        except (ValueError, TypeError):
            pass
    return False


def _mean(lst: list) -> float:
    return round(sum(lst) / max(len(lst), 1), 4)

def _f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  L1 — MOLECULE-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def score_molecule_pair(pred: dict, gt: dict) -> dict:
    """
    Score one predicted extraction against its matched GT row.

    Returns
    -------
    {
      property_accuracy : 0 or 1
      value_accuracy    : 0 or 1
      unit_accuracy     : 0 or 1
      tuple_integrity   : 0 or 1  (all three simultaneously correct + name match)
    }
    """
    prop_ok  = int(property_match(pred.get("property", ""),
                                  gt.get("property_label", "")))
    val_ok   = int(value_match(pred.get("value", ""), gt.get("value", "")))
    unit_ok  = int(units_match(pred.get("unit", ""), gt.get("unit", "")))

    tuple_ok = int(
        name_match(
            pred.get("chemical_name", ""), gt.get("chemical_name", ""),
            pred.get("chemical_name_long", ""), gt.get("chemical_name_long", ""),
        )
        and prop_ok and val_ok and unit_ok
    )

    return {
        "property_accuracy": prop_ok,
        "value_accuracy":    val_ok,
        "unit_accuracy":     unit_ok,
        "tuple_integrity":   tuple_ok,
    }


def match_pred_to_gt(pred: dict, gts: list[dict]) -> Optional[dict]:
    """
    Find the best GT row for a prediction.

    Primary anchor  : same chemical name AND same property type.
                      When multiple GTs share the same name+property (e.g. same
                      property reported in two different units), prefer the one
                      whose value also matches; otherwise take the first.
    Fallback anchor : same property type only — used exclusively when the
                      document has a single compound (all GT rows share the
                      same chemical name), preventing cross-compound leakage.
    """
    pred_name      = pred.get("chemical_name", "")
    pred_name_long = pred.get("chemical_name_long", "")
    pred_prop      = pred.get("property", "")
    pred_val       = pred.get("value", "")

    # Collect all name+property matches first
    candidates = [
        gt for gt in gts
        if name_match(pred_name, gt.get("chemical_name", ""),
                      pred_name_long, gt.get("chemical_name_long", ""))
        and property_match(pred_prop, gt.get("property_label", ""))
    ]

    if candidates:
        # Prefer a candidate whose value also matches (handles same prop in two units)
        for gt in candidates:
            if value_match(pred_val, gt.get("value", "")):
                return gt
        return candidates[0]

    # Fallback: property only — single-compound document where name roughly matches
    gt_names = {_norm_name(gt.get("chemical_name", "")) for gt in gts}
    if len(gt_names) == 1:
        single_name    = next(iter(gt_names))
        pred_name_norm = _norm_name(pred_name)
        if pred_name_norm and (pred_name_norm in single_name or single_name in pred_name_norm):
            for gt in gts:
                if property_match(pred_prop, gt.get("property_label", "")):
                    return gt

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  HALLUCINATION JUDGE  (LLM-as-Eval)
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: hallucination judging is handled entirely by evaluation/hallucination_check.py
# Run it as a separate step:
#   python evaluation/hallucination_check.py --exp A3 --judge-model <model>


# ══════════════════════════════════════════════════════════════════════════════
#  L2 — DOCUMENT-LEVEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_document(
    preds:       list[dict],
    gts:         list[dict],
    source_text: str = "",
) -> dict:
    """
    Full document-level evaluation.

    Parameters
    ----------
    preds             : predicted extractions
    gts               : ground-truth extractions (may be incomplete)
    source_text       : raw text (needed for hallucination judge)

    Key design choices
    ------------------
    - Predictions with no GT match are NOT counted as false positives.
      They are "novel" claims. The hallucination judge independently assesses
      whether they are grounded in the source text.
    - Recall / coverage is computed only over GT rows (lower bound; true
      recall is higher because GT is incomplete).
    - Field-level scores (property, value, unit, dependency) are computed
      only over matched (name+property) pairs.
    """
    if not gts and not preds:
        return _empty_doc_metrics()

    # ── Match predictions to GT ────────────────────────────────────────────
    mol_scores: list[dict]   = []   # L1 scores for matched pairs
    matched_gt_ids: set[int] = set()
    unverified_preds: list[dict] = []   # preds with no GT anchor

    for pred in preds:
        gt = match_pred_to_gt(pred, gts)
        if gt is not None:
            s = score_molecule_pair(pred, gt)
            s["pred"] = pred
            s["gt"]   = gt
            mol_scores.append(s)
            matched_gt_ids.add(id(gt))
        else:
            unverified_preds.append(pred)

    # ── Missed GT rows ─────────────────────────────────────────────────────
    missed_gts = [gt for gt in gts if id(gt) not in matched_gt_ids]

    # ── L1 aggregates ─────────────────────────────────────────────────────
    n_matched  = len(mol_scores)
    n_gt       = len(gts)
    n_pred     = len(preds)
    n_novel    = len(unverified_preds)
    n_missed   = len(missed_gts)

    def _field_mean(field: str, filter_fn=None):
        vals = [s[field] for s in mol_scores
                if s[field] is not None
                and (filter_fn is None or filter_fn(s))]
        return _mean(vals) if vals else None  # None = no matched pairs; excluded from global avg

    prop_acc  = _field_mean("property_accuracy")
    val_acc   = _field_mean("value_accuracy")
    unit_acc  = _field_mean("unit_accuracy")
    tuple_int = _field_mean("tuple_integrity")

    # ── M4: Extraction completeness (GT recall, no penalty for novel preds) ──
    # property_accuracy / value_accuracy / unit_accuracy are computed ONLY over
    # matched pairs (pred anchored to a GT row). Novel predictions (no GT match)
    # are NOT counted as errors here — they go to hallucination check separately.
    extraction_completeness = round(n_matched / max(n_gt, 1), 4)

    # ── M5: Tuple integrity score ──────────────────────────────────────────
    # Already computed above as fraction of matched pairs where all fields OK.

    # ── M6: Novel extraction rate ──────────────────────────────────────────
    # What fraction of all predictions are NOT in GT?
    # High value = finding things GT missed (good) or hallucinating (bad).
    # The hallucination judge disambiguates.
    novel_rate = round(n_novel / max(n_pred, 1), 4)

    # ── Hallucination placeholder ──────────────────────────────────────────
    # Full hallucination judging is done separately via:
    #   python evaluation/hallucination_check.py --exp <EXP>
    # Here we only count and flag unverified predictions structurally.
    halluc_results: list[dict] = [{**p, "verdict": "unverified"} for p in unverified_preds]
    n_grounded       = 0
    n_hallucinated   = 0
    confirmed_halluc_rate = 0.0

    return {
        # ── Counts ──────────────────────────────────────────────────────────
        "n_pred":     n_pred,
        "n_gt":       n_gt,
        "n_matched":  n_matched,
        "n_missed":   n_missed,
        "n_novel":    n_novel,

        # ── L1: Molecule-level field accuracy ───────────────────────────────
        "L1": {
            "property_accuracy": round(prop_acc,  4) if prop_acc  is not None else None,
            "value_accuracy":    round(val_acc,   4) if val_acc   is not None else None,
            "unit_accuracy":     round(unit_acc,  4) if unit_acc  is not None else None,
            "tuple_integrity":   round(tuple_int, 4) if tuple_int is not None else None,
        },

        # ── L2: Document-level aggregates ────────────────────────────────────
        "L2": {
            "mean_property_accuracy":  round(prop_acc,  4) if prop_acc  is not None else None,
            "mean_value_accuracy":     round(val_acc,   4) if val_acc   is not None else None,
            "mean_unit_accuracy":      round(unit_acc,  4) if unit_acc  is not None else None,
            "extraction_completeness": extraction_completeness,
            "tuple_integrity_score":   round(tuple_int, 4) if tuple_int is not None else None,
            "novel_extraction_rate":   novel_rate,
        },

        # ── Hallucination ────────────────────────────────────────────────────
        "hallucination": {
            "n_unverified":             len(unverified_preds),
            "n_grounded":               n_grounded,
            "n_hallucinated":           n_hallucinated,
            "confirmed_halluc_rate":    confirmed_halluc_rate,
            "judge_ran":                False,
            "verdicts":                 halluc_results,
        },

        # ── Detail lists ─────────────────────────────────────────────────────
        "missed_gts":     missed_gts,
        "mol_scores":     [{k: v for k, v in s.items() if k not in ("pred","gt")}
                           for s in mol_scores],
    }


def _empty_doc_metrics() -> dict:
    return {
        "n_pred": 0, "n_gt": 0, "n_matched": 0, "n_missed": 0, "n_novel": 0,
        "L1": {
            "property_accuracy": 0, "value_accuracy": 0,
            "unit_accuracy": 0, "tuple_integrity": 0,
        },
        "L2": {
            "mean_property_accuracy": 0, "mean_value_accuracy": 0,
            "mean_unit_accuracy": 0,
            "extraction_completeness": 0, "tuple_integrity_score": 0,
            "novel_extraction_rate": 0,
        },
        "hallucination": {
            "n_unverified": 0, "n_grounded": 0, "n_hallucinated": 0,
            "confirmed_halluc_rate": 0, "judge_ran": False, "verdicts": [],
        },
        "missed_gts": [], "mol_scores": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE ACROSS DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_results(
    run_results: list,
    dataset:     dict,
) -> dict:
    """
    Aggregate all per-document metrics across a full experiment run.

    Parameters
    ----------
    """
    # Build GT lookup keyed by text_id (doi#example_id) for multi-text DOI support.
    # Falls back to doi if doc has no text_id attribute.
    gt_by_tid: dict[str, list] = defaultdict(list)
    text_by_tid: dict[str, str] = {}
    docs_by_tid: dict[str, object] = {}
    for doc in dataset.get("documents", []):
        tid = getattr(doc, "text_id", None) or doc.doi
        for e in doc.extractions:
            gt_by_tid[tid].append(e.to_dict())
        text_by_tid[tid] = doc.text
        docs_by_tid[tid] = doc

    doc_metrics: list[dict] = []
    by_tier:     dict[str, list] = defaultdict(list)
    by_prop:     dict[int, list] = defaultdict(list)

    for rr in run_results:
        preds = rr.extractions if hasattr(rr, "extractions") else rr.get("extractions", [])
        tid   = rr.doc_id      if hasattr(rr, "doc_id")      else rr.get("doc_id", "")
        gts   = gt_by_tid.get(tid, [])
        text  = text_by_tid.get(tid, "")

        dm = evaluate_document(preds, gts, source_text=text)
        dm["doc_id"]  = tid
        dm["latency"] = rr.latency_seconds if hasattr(rr, "latency_seconds") else 0
        dm["iters"]   = (rr.reflection_stats.get("iters_used", 0)
                         if hasattr(rr, "reflection_stats") else 0)
        doc_metrics.append(dm)

        # Per-property extraction completeness
        for gt in gts:
            pid = gt.get("prop_id")
            if pid:
                found = any(
                    name_match(p.get("chemical_name",""), gt.get("chemical_name",""))
                    and property_match(p.get("property",""), gt.get("property_label",""))
                    for p in preds
                )
                by_prop[pid].append(1.0 if found else 0.0)

        # Per-tier
        doc = docs_by_tid.get(tid)
        if doc:
            for pid in doc.prop_ids:
                tier = config.PROPERTY_TIER.get(pid, "unknown")
                by_tier[tier].append(dm)

    # ── L1 global averages ─────────────────────────────────────────────────
    def _dm_mean(field: str) -> float:
        vals = [dm["L1"].get(field, 0) for dm in doc_metrics
                if dm["L1"].get(field) is not None]
        return _mean(vals) if vals else 0.0

    # ── L2 global averages ─────────────────────────────────────────────────
    def _l2(field: str) -> float:
        vals = [dm["L2"].get(field) for dm in doc_metrics if dm["L2"].get(field) is not None]
        return _mean(vals) if vals else 0.0

    # ── Tier-level L2 scores ───────────────────────────────────────────────
    def _tier_l2(mets, field: str) -> float:
        vals = [m["L2"].get(field) for m in mets if m["L2"].get(field) is not None]
        return _mean(vals) if vals else 0.0

    tier_scores: dict[str, dict] = {}
    for tier, mets in by_tier.items():
        tier_scores[tier] = {
            "extraction_completeness":  _tier_l2(mets, "extraction_completeness"),
            "tuple_integrity_score":    _tier_l2(mets, "tuple_integrity_score"),
            "mean_value_accuracy":      _tier_l2(mets, "mean_value_accuracy"),
            "novel_extraction_rate":    _tier_l2(mets, "novel_extraction_rate"),
            "confirmed_halluc_rate":    _mean([m["hallucination"]["confirmed_halluc_rate"] for m in mets]),
        }

    # ── Hallucination global ───────────────────────────────────────────────
    total_novel     = sum(dm["hallucination"]["n_unverified"]   for dm in doc_metrics)
    total_grounded  = sum(dm["hallucination"]["n_grounded"]     for dm in doc_metrics)
    total_halluc    = sum(dm["hallucination"]["n_hallucinated"]  for dm in doc_metrics)
    total_pred      = sum(dm["n_pred"]                          for dm in doc_metrics)
    judge_ran       = any(dm["hallucination"]["judge_ran"]      for dm in doc_metrics)

    return {
        "summary": {
            "n_documents":  len(doc_metrics),
            "total_gt":     sum(dm["n_gt"]   for dm in doc_metrics),
            "total_pred":   total_pred,
            "total_matched":sum(dm["n_matched"] for dm in doc_metrics),
            "total_novel":  total_novel,
        },

        # ── L1: Molecule-level (global averages) ───────────────────────────
        "L1_global": {
            "property_accuracy": _dm_mean("property_accuracy"),
            "value_accuracy":    _dm_mean("value_accuracy"),
            "unit_accuracy":     _dm_mean("unit_accuracy"),
            "tuple_integrity":   _dm_mean("tuple_integrity"),
        },

        # ── L2: Document-level (global averages) ───────────────────────────
        "L2_global": {
            "extraction_completeness": _l2("extraction_completeness"),
            "tuple_integrity_score":   _l2("tuple_integrity_score"),
            "mean_property_accuracy":  _l2("mean_property_accuracy"),
            "mean_value_accuracy":     _l2("mean_value_accuracy"),
            "mean_unit_accuracy":      _l2("mean_unit_accuracy"),
            "novel_extraction_rate":   _l2("novel_extraction_rate"),
        },

        # ── Hallucination ──────────────────────────────────────────────────
        "hallucination": {
            "judge_ran":              judge_ran,
            "total_novel_claims":     total_novel,
            "total_grounded":         total_grounded,
            "total_confirmed_halluc": total_halluc,
            "global_confirmed_halluc_rate": round(total_halluc / max(total_pred, 1), 4),
            "global_novel_rate":      round(total_novel / max(total_pred, 1), 4),
            "grounded_of_novel_rate": round(total_grounded / max(total_novel, 1), 4),
        },

        # ── Generalization ─────────────────────────────────────────────────
        "generalization": {
            "tier_scores":     tier_scores,
            "property_completeness": {
                str(k): _mean(v) for k, v in by_prop.items()
            },
        },

        # ── Efficiency ─────────────────────────────────────────────────────
        "efficiency": {
            "mean_reflection_iters": _mean([dm.get("iters", 0)   for dm in doc_metrics]),
            "mean_latency_seconds":  _mean([dm.get("latency", 0) for dm in doc_metrics]),
        },

        "per_document": doc_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BASE ACCURACY  (GT-anchored; kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def base_accuracy(preds: list[dict], gts: list[dict]) -> float:
    """Fraction of GT rows hit by any prediction (name + property + value match)."""
    if not gts:
        return 1.0
    hits = sum(
        1 for gt in gts
        if any(
            name_match(p.get("chemical_name",""), gt.get("chemical_name",""))
            and property_match(p.get("property",""), gt.get("property_label",""))
            and value_match(p.get("value",""), gt.get("value",""))
            for p in preds
        )
    )
    return round(hits / len(gts), 4)


def aggregate_base_accuracy(run_results: list, dataset: dict) -> dict:
    """Base accuracy broken out by property, covariate type, name type, tier."""
    gt_by_tid: dict[str, list] = defaultdict(list)
    for doc in dataset.get("documents", []):
        tid = getattr(doc, "text_id", None) or doc.doi
        for e in doc.extractions:
            gt_by_tid[tid].append(e.to_dict())

    overall:        list[float] = []
    by_prop:        dict[int, list]   = defaultdict(list)
    by_covariate:   dict[str, list]   = {"with_covariate": [], "no_covariate": []}
    by_name_type:   dict[str, list]   = defaultdict(list)
    by_tier:        dict[str, list]   = defaultdict(list)

    for rr in run_results:
        preds = rr.extractions if hasattr(rr, "extractions") else rr.get("extractions", [])
        tid   = rr.doc_id      if hasattr(rr, "doc_id")      else rr.get("doc_id", "")
        gts   = gt_by_tid.get(tid, [])

        for gt in gts:
            hit = any(
                name_match(p.get("chemical_name",""), gt.get("chemical_name",""))
                and property_match(p.get("property",""), gt.get("property_label",""))
                and value_match(p.get("value",""), gt.get("value",""))
                for p in preds
            )
            score = 1.0 if hit else 0.0
            overall.append(score)

            pid = gt.get("prop_id")
            if pid:
                by_prop[pid].append(score)
                tier = config.PROPERTY_TIER.get(pid, "unknown")
                by_tier[tier].append(score)

            cov_key = "with_covariate" if gt.get("covariate_value") else "no_covariate"
            by_covariate[cov_key].append(score)
            by_name_type[str(gt.get("name_type", "unknown")).lower()].append(score)

    return {
        "overall":            _mean(overall),
        "by_property":        {str(k): _mean(v) for k, v in sorted(by_prop.items())},
        "by_covariate":       {k: _mean(v) for k, v in by_covariate.items()},
        "by_name_type":       {k: _mean(v) for k, v in by_name_type.items()},
        "by_name_type_counts":{k: len(v)   for k, v in by_name_type.items()},
        "by_tier":            {k: _mean(v) for k, v in by_tier.items()},
        "n_gt_total":         len(overall),
        "n_gt_hit":           int(sum(overall)),
    }
