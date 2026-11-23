#!/usr/bin/env python3
"""
experiments/run_baselines.py — B0 (zero-shot) and B1 (1-shot).

Both baselines use the full 3-agent pipeline (Detector → Extractor → Reflector).
B0 passes no examples to any agent; B1 passes one training example to each.

Usage:
  python experiments/run_baselines.py
  python experiments/run_baselines.py --exp B0
  python experiments/run_baselines.py --max-docs 20
  python experiments/run_baselines.py --redo B0
  python experiments/run_baselines.py --resume
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.loader import load_dataset
from data.index import PropertyIndex
from agents.pipeline import run_experiment
from evaluation.metrics import aggregate_results, aggregate_base_accuracy
from utils.error_store import ErrorStore, merge_redo_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(i) for i in o]
    if isinstance(o, float) and (o != o or o == float("inf") or o == float("-inf")): return None
    return o

def _load_results(path) -> list:
    """Load existing results JSON, returning [] on missing or empty/corrupt file."""
    if not path.exists():
        return []
    raw = path.read_text().strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[_load_results] Corrupt JSON in {path.name} — treating as empty.")
        return []

def _append_results(path, new_results):
    existing = _load_results(path)
    ids = {r.get("doc_id") for r in existing}
    for rr in new_results:
        if rr.doc_id not in ids:
            existing.append(_clean(rr.to_result_dict() if hasattr(rr,"to_result_dict") else rr.to_dict()))
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def _repair_checkpoint(exp_id, results_dir, logs_dir=None):
    """
    Cross-check _completed.txt against _results.json.

    For any doc that is checkpointed but missing from the results JSON,
    recover its result from logs/<EXP>_docs.jsonl or logs/<EXP>_full_trace.jsonl
    — NO API calls are made.  Only docs that cannot be found in ANY log file
    are un-marked from the checkpoint and scheduled for re-run.
    """
    from utils.recover import repair_checkpoint_with_recovery
    if logs_dir is None:
        logs_dir = config.get_logs_dir()
    repair_checkpoint_with_recovery(exp_id, results_dir, logs_dir)

def run_one(exp_id, data, max_docs, results_dir, logs_dir, resume=False):
    if resume:
        _repair_checkpoint(exp_id, results_dir, logs_dir)
    good, failed = run_experiment(
        documents=data["test_docs"], experiment_id=exp_id,
        train_docs=data["train_docs"],
        max_docs=max_docs, logs_dir=logs_dir,
        results_dir=results_dir, skip_completed=resume,
    )
    if not good:
        logger.warning(f"[{exp_id}] No successful results.")
        return
    metrics  = aggregate_results(good, data)
    base_acc = aggregate_base_accuracy(good, data)
    _append_results(results_dir / f"{exp_id}_results.json", good)
    (results_dir / f"{exp_id}_metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    (results_dir / f"{exp_id}_base_accuracy.json").write_text(json.dumps(_clean(base_acc), indent=2), encoding="utf-8")
    l2 = metrics.get("L2_global", {})
    logger.info(f"[{exp_id}] Compl={l2.get('extraction_completeness',0):.4f}  BaseAcc={base_acc['overall']:.4f}  "
                f"failed={len(failed)}  "
                + (f"| run: python experiments/run_baselines.py --redo {exp_id}" if failed else ""))

def redo_one(exp_id, data, results_dir, logs_dir, max_docs=None):
    from agents.pipeline import run_pipeline
    store = ErrorStore(results_dir, exp_id)
    failed_ids = store.load_failed()
    if not failed_ids:
        logger.info(f"[Redo] {exp_id}: nothing to redo."); return
    # doc_map keyed by text_id (doi#example_id) with fallback to doi
    doc_map = {}
    for d in data["documents"]:
        tid = getattr(d, "text_id", None) or d.doi
        doc_map[tid] = d
    cfg = config.EXPERIMENTS.get(exp_id, {})
    static_pool = []
    if cfg.get("use_examples", False):
        for doc in data["train_docs"]:
            for e in doc.extractions:
                static_pool.append({"source":"retrieved","text":doc.text[:400],"extraction":e.to_dict()})
    good_redo = []
    for did in (failed_ids[:max_docs] if max_docs else failed_ids):
        doc = doc_map.get(did)
        if not doc: continue
        res = run_pipeline(doc.text, did, exp_id,
                           static_pool=static_pool, logs_dir=logs_dir, error_store=store)
        if not res.failed:
            good_redo.append(res); store.clear_failed(did)
    if good_redo:
        merge_redo_results(results_dir / f"{exp_id}_results.json", good_redo, store)
    logger.info(f"[Redo] {exp_id}: {len(good_redo)}/{len(failed_ids)} recovered")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",      default="all", help="B0|B1|all")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--redo",     nargs="*", metavar="EXP_ID",
                   help="Re-run failed docs for given IDs (or 'all')")
    p.add_argument("--resume",   action="store_true")
    args = p.parse_args()

    data        = load_dataset()
    results_dir = config.get_results_dir()
    logs_dir    = config.get_logs_dir()

    baseline_ids = ["B0", "B1"]

    if args.redo is not None:
        ids = args.redo if args.redo and args.redo != ["all"] else baseline_ids
        for eid in ids:
            redo_one(eid, data, results_dir, logs_dir, args.max_docs)
        return

    exps = baseline_ids if args.exp == "all" else [args.exp]
    for exp_id in exps:
        run_one(exp_id, data, args.max_docs, results_dir, logs_dir, args.resume)

if __name__ == "__main__":
    main()
