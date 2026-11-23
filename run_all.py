#!/usr/bin/env python3
"""
run_all.py — Master orchestrator with error handling and --redo support.

Output layout  (per extractor model):
  results/<model>/
    <EXP>_results.json         successful extractions only
    <EXP>_metrics.json         aggregated metrics
    <EXP>_base_accuracy.json   base accuracy breakdown
    <EXP>_errors.jsonl         one record per failed document (full trace)
    <EXP>_failed_ids.txt       one doc_id per line  ← feed to --redo
    <EXP>_completed.txt        checkpoint (skip on resume)
    logs/<EXP>_docs.jsonl      every document, good or bad, full trace
    run_all.log                console mirror
  figures/<model>/
    fig*.png

Usage:
  python run_all.py                          # full run, all experiments
  python run_all.py --max-docs 10            # quick test
  python run_all.py --only B1               # one experiment
  python run_all.py --skip B0               # exclude experiments
  python run_all.py --figures-only
  python run_all.py --redo B1               # re-run failed docs for B1
  python run_all.py --redo B0 B1            # re-run failed docs for multiple
  python run_all.py --redo all              # re-run failed docs for every exp
  python run_all.py --resume                # skip already-completed docs
"""

import sys
import json
import argparse
import logging
from collections import defaultdict
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent))

import config
from data.loader import load_dataset, save_dataset_json
from data.index import PropertyIndex
from agents.pipeline import run_experiment, run_pipeline, RunResult
from evaluation.metrics import aggregate_results, aggregate_base_accuracy
from evaluation.hallucination_check import run_hallucination_check
from figures.visualize import generate_all_figures
from utils.error_store import ErrorStore, merge_redo_results


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_all.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )


logger = logging.getLogger(__name__)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _clean(obj):
    if isinstance(obj, dict):           return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  return [_clean(i) for i in obj]
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None   # NaN / Inf — not valid JSON
    return obj


def _append_results(results_file: Path, good: list[RunResult]):
    """
    Append clean extraction records to <EXP>_results.json.
    Uses to_result_dict() — extractions only, no LLM call internals.
    Full trace (with LLM calls) is in logs/<EXP>_full_trace.jsonl.
    """
    existing = []
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing_ids = {r.get("doc_id") for r in existing}
    for rr in good:
        if rr.doc_id not in existing_ids:
            d = rr.to_result_dict() if hasattr(rr, "to_result_dict") else _clean(rr.to_dict())
            existing.append(_clean(d))
    results_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── Save experiment outputs ───────────────────────────────────────────────────

def save_experiment(
    exp_id:      str,
    good:        list[RunResult],
    metrics:     dict,
    base_acc:    dict,
    results_dir: Path,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{exp_id}_results.json"
    _append_results(results_file, good)
    (results_dir / f"{exp_id}_metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    (results_dir / f"{exp_id}_base_accuracy.json").write_text(
        json.dumps(_clean(base_acc), indent=2), encoding="utf-8")
    logger.info(
        f"[{exp_id}] Saved {len(good)} results | "
        f"BaseAcc={base_acc['overall']:.4f} | → {results_dir.name}/"
    )


# ── Core experiment ───────────────────────────────────────────────────────────

def _repair_checkpoint(exp_id: str, results_dir: Path, logs_dir: Path = None):
    """
    Cross-check _completed.txt against _results.json before a --resume run.

    For any doc that is checkpointed but missing from the results JSON,
    recover its result from logs/<EXP>_docs.jsonl or logs/<EXP>_full_trace.jsonl
    — NO API calls are made.  Only docs that cannot be found in ANY log file
    are un-marked from the checkpoint and scheduled for re-run.
    """
    from utils.recover import repair_checkpoint_with_recovery
    if logs_dir is None:
        logs_dir = config.get_logs_dir()
    repair_checkpoint_with_recovery(exp_id, results_dir, logs_dir)


def run_single(
    exp_id:      str,
    data:        dict,
    index,
    max_docs:    int,
    results_dir: Path,
    logs_dir:    Path,
    skip_completed: bool = False,
) -> tuple[dict, dict]:
    if skip_completed:
        _repair_checkpoint(exp_id, results_dir, logs_dir)

    logger.info("\n" + "-"*60)
    logger.info(f"  {exp_id}: {config.EXPERIMENTS[exp_id]['name']}")
    logger.info(f"  extractor: {config.MODELS['extractor']} | backend: {config.LLM_BACKEND}")
    logger.info("-"*60)

    good, failed = run_experiment(
        documents=data["test_docs"],
        experiment_id=exp_id,
        index=index,
        train_docs=data["train_docs"],
        max_docs=max_docs,
        logs_dir=logs_dir,
        results_dir=results_dir,
        skip_completed=skip_completed,
    )

    if not good:
        logger.warning(f"[{exp_id}] No successful results — skipping metrics.")
        return {}, {}

    metrics  = aggregate_results(good, data)
    base_acc = aggregate_base_accuracy(good, data)
    save_experiment(exp_id, good, metrics, base_acc, results_dir)

    if failed:
        store = ErrorStore(results_dir, exp_id)
        logger.warning(
            f"[{exp_id}] {len(failed)} document(s) FAILED. "
            f"Error breakdown: {store.summary()['error_kinds']}. "
            f"Re-run with: python run_all.py --redo {exp_id}"
        )

    el = metrics.get("L2_global", {})
    logger.info(
        f"[{exp_id}] Compl={el.get('extraction_completeness',0):.4f}  "
        f"ValAcc={el.get('mean_value_accuracy',0):.4f}  "
        f"TupleInt={el.get('tuple_integrity_score',0):.4f}  "
        f"Novel={el.get('novel_extraction_rate',0):.4f}  "
        f"BaseAcc={base_acc.get('overall',0):.4f}"
    )
    return metrics, base_acc


# ── Redo failed documents ─────────────────────────────────────────────────────

def redo_experiment(
    exp_id:      str,
    data:        dict,
    index,
    results_dir: Path,
    logs_dir:    Path,
    max_docs:    int = None,
) -> tuple[dict, dict]:
    """Re-run only the documents listed in <EXP>_failed_ids.txt."""
    store      = ErrorStore(results_dir, exp_id)
    failed_ids = store.load_failed()

    if not failed_ids:
        logger.info(f"[Redo] {exp_id}: no failed documents on record.")
        return {}, {}

    logger.info("\n" + "-"*60)
    logger.info(f"  REDO {exp_id}: {len(failed_ids)} document(s)")
    logger.info("-"*60)

    # Build a lookup from doi → Document
    doc_map = {d.doi: d for d in data["documents"]}
    to_redo  = [doc_map[did] for did in failed_ids if did in doc_map]
    missing  = [did for did in failed_ids if did not in doc_map]
    if missing:
        logger.warning(f"[Redo] {exp_id}: {len(missing)} doc_id(s) not found in dataset.")

    if not to_redo:
        logger.warning(f"[Redo] {exp_id}: nothing to redo after filtering.")
        return {}, {}

    if max_docs:
        to_redo = to_redo[:max_docs]

    cfg = config.EXPERIMENTS.get(exp_id, config.EXPERIMENTS["B1"])
    static_pool = []
    if cfg.get("use_examples", False):
        for doc in data["train_docs"]:
            for e in doc.extractions:
                static_pool.append({
                    "source": "retrieved",
                    "text": doc.text[:400],
                    "extraction": e.to_dict(),
                })

    good_redo: list[RunResult] = []
    still_failed: list[RunResult] = []

    for i, doc in enumerate(to_redo):
        logger.info(f"  Redo {i+1}/{len(to_redo)}: {doc.doi[:50]}")
        res = run_pipeline(
            text=doc.text,
            doc_id=doc.doi,
            experiment_id=exp_id,
            index=index,
            static_pool=static_pool,
            logs_dir=logs_dir,
            error_store=store,
        )
        if res.failed:
            still_failed.append(res)
        else:
            good_redo.append(res)
            store.clear_failed(doc.doi)

    logger.info(
        f"[Redo] {exp_id}: {len(good_redo)} recovered, "
        f"{len(still_failed)} still failing"
    )

    if not good_redo:
        return {}, {}

    # Merge into main results file
    results_file = results_dir / f"{exp_id}_results.json"
    merge_redo_results(results_file, good_redo, store)

    # Recompute metrics over the full (old + redo) results
    all_good = _load_all_good(results_file)
    if not all_good:
        return {}, {}

    metrics  = aggregate_results(all_good, data)
    base_acc = aggregate_base_accuracy(all_good, data)
    (results_dir / f"{exp_id}_metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    (results_dir / f"{exp_id}_base_accuracy.json").write_text(
        json.dumps(_clean(base_acc), indent=2), encoding="utf-8")

    el = metrics.get("L2_global", {})
    logger.info(
        f"[Redo] {exp_id} updated — Compl={el.get('extraction_completeness',0):.4f}  "
        f"BaseAcc={base_acc.get('overall',0):.4f}"
    )
    return metrics, base_acc


def _load_all_good(results_file: Path) -> list:
    """Load all records from the main results JSON as lightweight dicts."""
    if not results_file.exists():
        return []
    try:
        records = json.loads(results_file.read_text(encoding="utf-8"))
    except Exception:
        return []

    # Wrap in lightweight objects compatible with aggregate_results / merge_redo_results
    class _Rec:
        def __init__(self, d):
            self.__dict__.update(d)
            if not hasattr(self, "extractions"):       self.extractions = []
            if not hasattr(self, "doc_id"):            self.doc_id = ""
            if not hasattr(self, "latency_seconds"):   self.latency_seconds = 0.0
            if not hasattr(self, "reflection_stats"):  self.reflection_stats = {}
        def to_dict(self):        return self.__dict__
        def to_result_dict(self): return self.__dict__   # already slim
        @property
        def failed(self): return self.__dict__.get("status", "ok") == "failed"

    return [_Rec(r) for r in records if r.get("status","ok") != "failed"]


# ── Hallucination check helpers ───────────────────────────────────────────────

def _patch_metrics_halluc(exp_id: str, results_dir: Path, summary: dict) -> dict:
    """Merge real hallucination numbers into the saved metrics JSON; return updated dict."""
    m_path = results_dir / f"{exp_id}_metrics.json"
    if not m_path.exists():
        return {}
    try:
        m = json.loads(m_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    halluc = m.get("hallucination", {})
    halluc.update({
        "judge_ran":                    True,
        "global_confirmed_halluc_rate": summary.get("halluc_rate_of_total", 0.0),
        "total_novel_claims":           summary.get("total_novel_claims", 0),
        "total_grounded":               summary.get("total_grounded", 0),
        "total_hallucinated":           summary.get("total_hallucinated", 0),
        "total_ambiguous":              summary.get("total_ambiguous", 0),
        "halluc_rate_of_novel":         summary.get("halluc_rate_of_novel", 0.0),
        "grounded_rate_of_novel":       summary.get("grounded_rate_of_novel", 0.0),
    })
    m["hallucination"] = halluc
    m_path.write_text(json.dumps(_clean(m), indent=2), encoding="utf-8")
    return m


def run_halluc_checks(exp_ids: list[str], results_dir: Path, data: dict) -> dict:
    """
    Run LLM-as-judge hallucination check for each experiment that has saved results.
    Updates each *_metrics.json with real hallucination rates so print_table is accurate.
    Returns {exp_id: updated_metrics_dict}.
    """
    updated: dict[str, dict] = {}
    for exp_id in exp_ids:
        if not (results_dir / f"{exp_id}_results.json").exists():
            continue
        try:
            summary = run_hallucination_check(
                exp_id        = exp_id,
                results_dir   = results_dir,
                dataset       = data,
            )
            if summary:
                m = _patch_metrics_halluc(exp_id, results_dir, summary)
                if m:
                    updated[exp_id] = m
        except Exception as exc:
            logger.warning(f"[{exp_id}] Hallucination check failed: {exc}")
    return updated


# ── Summary table ─────────────────────────────────────────────────────────────

def _m(metrics, *keys, default=0):
    """Safely walk nested dicts."""
    obj = metrics
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj if obj is not None else default


def print_table(exp_metrics: dict, base_accs: dict, results_dir: Path):
    exps = [e for e in ["B0", "B1"] if e in exp_metrics]
    if not exps:
        return

    w = 90
    print(f"\n{'='*w}")
    print(f"{'Exp':<6} {'Compl':>7} {'PropAcc':>8} {'ValAcc':>8} {'UnitAcc':>8} "
          f"{'TupleInt':>9} {'NovRate':>8} {'HallucRate':>11} {'Failed':>7}")
    print("-"*w)
    for exp_id in exps:
        m      = exp_metrics[exp_id]
        store  = ErrorStore(results_dir, exp_id)
        n_fail = len(store.load_failed())
        print(
            f"{exp_id:<6}"
            f" {_m(m,'L2_global','extraction_completeness'):>7.4f}"
            f" {_m(m,'L2_global','mean_property_accuracy'):>8.4f}"
            f" {_m(m,'L2_global','mean_value_accuracy'):>8.4f}"
            f" {_m(m,'L2_global','mean_unit_accuracy'):>8.4f}"
            f" {_m(m,'L2_global','tuple_integrity_score'):>9.4f}"
            f" {_m(m,'L2_global','novel_extraction_rate'):>8.4f}"
            f" {_m(m,'hallucination','global_confirmed_halluc_rate'):>11.4f}"
            f" {n_fail:>7}"
        )
    print("="*w)

    # ── Tier breakdown ────────────────────────────────────────────────────
    for exp_id in exps:
        tiers = _m(exp_metrics[exp_id], "generalization", "tier_scores", default={})
        if tiers:
            print(f"\n{exp_id} Generalization by Tier:")
            print(f"  {'Tier':<25} {'Compl':>7} {'ValAcc':>8} {'TupleInt':>9} {'Novel':>7}")
            print("  " + "-"*58)
            for t, ts in sorted(tiers.items()):
                print(
                    f"  {t:<25}"
                    f" {ts.get('extraction_completeness',0):>7.4f}"
                    f" {ts.get('mean_value_accuracy',0):>8.4f}"
                    f" {ts.get('tuple_integrity_score',0):>9.4f}"
                    f" {ts.get('novel_extraction_rate',0):>7.4f}"
                )

    if base_accs:
        print("\nBase Accuracy (GT-anchored recall):")
        for exp_id in exps:
            ba = base_accs.get(exp_id, {})
            print(f"  {exp_id:<6}  overall={ba.get('overall',0):.4f}  "
                  + "  ".join(f"{t}={v:.4f}" for t, v in ba.get("by_tier", {}).items()))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="OmniPropAgent master runner")
    p.add_argument("--max-docs",      type=int, default=None,
                   help="Limit test documents per experiment")
    p.add_argument("--only",          nargs="*", default=None,
                   help="Run only these experiment IDs")
    p.add_argument("--skip",          nargs="*", default=[],
                   help="Experiment IDs to skip")
    p.add_argument("--no-ablations",  action="store_true", help="(no-op; ablations removed)")
    p.add_argument("--figures-only",  action="store_true",
                   help="Regenerate figures from saved results")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--resume",        action="store_true",
                   help="Skip documents already in completed checkpoint")
    p.add_argument("--redo",          nargs="*", metavar="EXP_ID",
                   help="Re-run failed documents. Use 'all' to redo every experiment.")
    p.add_argument("--eval-only",     nargs="*", metavar="EXP_ID",
                   help="Re-evaluate existing predictions from trace (no new LLM calls). "
                        "Replaces _results.json, _metrics.json, _base_accuracy.json.")
    p.add_argument("--seed",          type=int, default=config.SEED)
    p.add_argument("--no-halluc",     action="store_true",
                   help="Skip the automatic hallucination check after evaluation.")
    args = p.parse_args()

    random.seed(args.seed)

    results_dir = config.get_results_dir()
    figures_dir = config.get_figures_dir()
    logs_dir    = config.get_logs_dir()
    setup_logging(results_dir)

    logger.info(f"Extractor model : {config.MODELS['extractor']}")
    logger.info(f"Detector model  : {config.MODELS['detector']}")
    logger.info(f"Reflector model : {config.MODELS['reflector']}")
    logger.info(f"Backend         : {config.LLM_BACKEND}")
    logger.info(f"Results dir     : {results_dir}")
    logger.info(f"Figures dir     : {figures_dir}")

    # ── Dataset & index ───────────────────────────────────────────────────
    data = load_dataset()
    save_dataset_json(data, results_dir / "dataset.json")

    # Index kept for figure generation only; not used in pipeline
    if args.rebuild_index:
        index = PropertyIndex(); index.build(data["train_docs"]); index.save()
    else:
        index = PropertyIndex.load_or_build(data["train_docs"])

    # ── Figures-only ──────────────────────────────────────────────────────
    if args.figures_only:
        generate_all_figures(results_dir=results_dir, figures_dir=figures_dir,
                             index=index, extra={})
        return

    # ── Eval-only mode — re-evaluate from trace, no LLM calls ────────────
    if args.eval_only is not None:
        from experiments.rebuild_results_from_trace import rebuild
        eval_ids = args.eval_only or list(config.EXPERIMENTS.keys())
        exp_metrics: dict = {}
        base_accs:   dict = {}
        for exp_id in eval_ids:
            rebuild(exp_id, data, results_dir, logs_dir)
            m_path  = results_dir / f"{exp_id}_metrics.json"
            ba_path = results_dir / f"{exp_id}_base_accuracy.json"
            if m_path.exists():
                exp_metrics[exp_id] = json.loads(m_path.read_text(encoding="utf-8"))
            if ba_path.exists():
                base_accs[exp_id]   = json.loads(ba_path.read_text(encoding="utf-8"))
        if not args.no_halluc:
            updated = run_halluc_checks(eval_ids, results_dir, data)
            exp_metrics.update(updated)
        if exp_metrics:
            print_table(exp_metrics, base_accs, results_dir)
        return

    # ── Redo mode ─────────────────────────────────────────────────────────
    if args.redo is not None:
        redo_ids = args.redo
        if not redo_ids or redo_ids == ["all"]:
            redo_ids = list(config.EXPERIMENTS.keys())

        exp_metrics: dict = {}
        base_accs:   dict = {}
        for exp_id in redo_ids:
            if exp_id not in config.EXPERIMENTS:
                logger.warning(f"Unknown experiment: {exp_id}"); continue
            m, ba = redo_experiment(exp_id, data, index, results_dir, logs_dir,
                                    max_docs=args.max_docs)
            if m:
                exp_metrics[exp_id] = m
                base_accs[exp_id]   = ba

        if not args.no_halluc and exp_metrics:
            updated = run_halluc_checks(list(exp_metrics.keys()), results_dir, data)
            exp_metrics.update(updated)
        if exp_metrics:
            print_table(exp_metrics, base_accs, results_dir)
            generate_all_figures(results_dir=results_dir, figures_dir=figures_dir,
                                 index=index, extra={})
        return


    # ── Normal run ────────────────────────────────────────────────────────
    exps_to_run = args.only or list(config.EXPERIMENTS.keys())
    exps_to_run = [e for e in exps_to_run
                   if e not in args.skip and e in config.EXPERIMENTS]

    exp_metrics: dict = {}
    base_accs:   dict = {}
    for exp_id in exps_to_run:
        m, ba = run_single(exp_id, data, index, args.max_docs,
                           results_dir, logs_dir,
                           skip_completed=args.resume)
        if m:
            exp_metrics[exp_id] = m
            base_accs[exp_id]   = ba

    # ── Hallucination check ───────────────────────────────────────────────
    if not args.no_halluc and exp_metrics:
        updated = run_halluc_checks(list(exp_metrics.keys()), results_dir, data)
        exp_metrics.update(updated)

    # ── Summary & figures ─────────────────────────────────────────────────
    if exp_metrics:
        print_table(exp_metrics, base_accs, results_dir)

    summary = {
        "extractor_model": config.MODELS["extractor"],
        "backend":         config.LLM_BACKEND,
        "experiments": {
            k: {"metrics": _clean(exp_metrics[k]),
                "base_accuracy": _clean(base_accs[k])}
            for k in exp_metrics
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    generate_all_figures(results_dir=results_dir, figures_dir=figures_dir,
                         index=index, extra={})

    logger.info(f"\nDone. Results → {results_dir}/  Figures → {figures_dir}/")


if __name__ == "__main__":
    main()
