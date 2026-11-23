#!/usr/bin/env python3
"""
experiments/rebuild_results_from_trace.py

Re-derives _results.json, _metrics.json, _base_accuracy.json from the
full trace JSONL.  Assigns each trace entry a text_id by matching
source_text against dataset.json, then re-evaluates against the correct
GT.

Usage:
  python experiments/rebuild_results_from_trace.py --exp B0
  python experiments/rebuild_results_from_trace.py --exp B0 B1
"""
import sys, json, argparse, logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.loader import load_dataset
from evaluation.metrics import aggregate_results, aggregate_base_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _clean(o):
    if isinstance(o, dict):  return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [_clean(i) for i in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


class _FakeRR:
    """Minimal RunResult-like object to feed aggregate_results."""
    def __init__(self, doc_id, extractions, reflection_stats, latency):
        self.doc_id           = doc_id
        self.extractions      = extractions
        self.reflection_stats = reflection_stats or {}
        self.latency_seconds  = latency or 0.0


def rebuild(exp_id: str, data: dict, results_dir: Path, logs_dir: Path):
    trace_path = logs_dir / f"{exp_id}_full_trace.jsonl"
    if not trace_path.exists():
        logger.error(f"Trace not found: {trace_path}")
        return

    # Build text -> text_id map from dataset
    text_to_tid: dict[str, str] = {}
    for doc in data["documents"]:
        text_to_tid[doc.text] = doc.text_id

    logger.info(f"[{exp_id}] Loading trace: {trace_path}")
    traces = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"[{exp_id}] {len(traces)} trace entries")

    results = []
    n_matched = 0
    n_fallback = 0
    n_missed = 0

    for t in traces:
        src_text = t.get("source_text", "")
        doi      = t.get("doc_id", "")

        # Try exact text match first
        tid = text_to_tid.get(src_text)
        if tid:
            n_matched += 1
        else:
            # Fallback: find doc with same DOI whose text is a prefix/suffix match
            for doc in data["documents"]:
                if doc.doi == doi and (
                    src_text.startswith(doc.text[:200]) or
                    doc.text.startswith(src_text[:200])
                ):
                    tid = doc.text_id
                    n_fallback += 1
                    break

        if not tid:
            # Last resort: use first doc with matching DOI
            for doc in data["documents"]:
                if doc.doi == doi:
                    tid = doc.text_id
                    n_missed += 1
                    logger.warning(f"  text_id fallback (doi only): {doi[:50]}")
                    break

        if not tid:
            logger.warning(f"  No match found for doc_id={doi[:50]}, skipping")
            continue

        exts       = t.get("final_extractions", [])
        ref_stats  = t.get("reflection_stats", {})
        latency    = t.get("latency_seconds", 0.0)

        results.append({
            "doc_id":           tid,
            "experiment_id":    t.get("experiment_id", exp_id),
            "status":           t.get("status", "ok"),
            "error":            t.get("error"),
            "error_kind":       t.get("error_kind"),
            "extractions":      exts,
            "n_extractions":    len(exts),
            "detected_props":   t.get("detected_props", []),
            "n_examples_used":  t.get("n_examples_used", 0),
            "reflection_stats": ref_stats,
            "latency_seconds":  latency,
        })

    logger.info(f"[{exp_id}] text_id resolution: {n_matched} exact | "
                f"{n_fallback} fallback | {n_missed} doi-only")
    logger.info(f"[{exp_id}] {len(results)} result entries to evaluate")

    # Save results JSON
    out_results = results_dir / f"{exp_id}_results.json"
    out_results.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"  Saved: {out_results}")

    # Evaluate
    fake_rrs = [
        _FakeRR(
            doc_id           = r["doc_id"],
            extractions      = r["extractions"],
            reflection_stats = r["reflection_stats"],
            latency          = r["latency_seconds"],
        )
        for r in results if r.get("status") == "ok"
    ]
    logger.info(f"[{exp_id}] Evaluating {len(fake_rrs)} successful docs...")

    metrics  = aggregate_results(fake_rrs, data)
    base_acc = aggregate_base_accuracy(fake_rrs, data)

    (results_dir / f"{exp_id}_metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / f"{exp_id}_base_accuracy.json").write_text(
        json.dumps(_clean(base_acc), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    l2 = metrics.get("L2_global", {})
    logger.info(
        f"[{exp_id}] Completeness={l2.get('extraction_completeness',0):.4f}  "
        f"TupleInt={l2.get('tuple_integrity_score',0):.4f}  "
        f"BaseAcc={base_acc['overall']:.4f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", default=["B0"])
    args = p.parse_args()

    data        = load_dataset()
    results_dir = config.get_results_dir()
    logs_dir    = config.get_logs_dir()

    for exp_id in args.exp:
        rebuild(exp_id, data, results_dir, logs_dir)


if __name__ == "__main__":
    main()
