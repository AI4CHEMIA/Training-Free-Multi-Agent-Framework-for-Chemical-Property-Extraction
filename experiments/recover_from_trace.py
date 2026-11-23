#!/usr/bin/env python3
"""
experiments/recover_from_trace.py — Recover *_results.json from log files.

Run this when a baseline experiment completed all API calls but crashed before
(or during) the final JSON write — e.g. a UnicodeEncodeError after all docs
were processed.  No new API calls are made.

Recovery reads from (tried in order):
  logs/<EXP>_docs.jsonl        ← exact to_result_dict() records, fastest
  logs/<EXP>_full_trace.jsonl  ← full trace, extractions under 'final_extractions'

After recovering the results JSON, the script re-computes and writes:
  <EXP>_metrics.json
  <EXP>_base_accuracy.json

Usage:
  python experiments/recover_from_trace.py --exp B0
  python experiments/recover_from_trace.py --exp B0 B1
  python experiments/recover_from_trace.py          # recovers all experiments that need it
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.recover import repair_checkpoint_with_recovery, _load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(i) for i in o]
    if isinstance(o, float) and (o != o or o == float("inf") or o == float("-inf")): return None
    return o


def _load_results(path: Path) -> list:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Corrupt JSON in {path.name} — will rebuild from logs.")
        return []


def rebuild_results_from_logs(exp_id: str, results_dir: Path, logs_dir: Path) -> list[dict]:
    """
    Load ALL docs from logs/<EXP>_docs.jsonl (or _full_trace.jsonl) and return
    a deduplicated list of result dicts — regardless of checkpoint state.
    Use this when _results.json is empty/corrupt but the log files are intact.
    """
    from utils.recover import _trace_to_result_dict

    results: dict[str, dict] = {}

    # Source 1: _docs.jsonl (already in to_result_dict() format)
    for rec in _load_jsonl(logs_dir / f"{exp_id}_docs.jsonl"):
        did = rec.get("doc_id", "")
        if did and did not in results:
            results[did] = rec

    # Source 2: _full_trace.jsonl (needs key mapping)
    for rec in _load_jsonl(logs_dir / f"{exp_id}_full_trace.jsonl"):
        did = rec.get("doc_id", "")
        if did and did not in results:
            results[did] = _trace_to_result_dict(rec)

    return list(results.values())


def compute_and_save_metrics(exp_id: str, result_dicts: list[dict],
                              results_dir: Path):
    """Re-compute metrics + base_accuracy from recovered result dicts."""
    # We need RunResult objects for aggregate_results; simulate with SimpleNamespace
    # Actually aggregate_results/aggregate_base_accuracy accept list of result_dicts
    # when called with dict-like objects.  Let's check what they actually need.
    try:
        from data.loader import load_dataset
        from evaluation.metrics import aggregate_results, aggregate_base_accuracy

        data = load_dataset()

        # Build lightweight proxy objects that expose the same interface
        # as RunResult but are constructed from plain dicts.
        class _Proxy:
            def __init__(self, d):
                self._d = d
                self.doc_id            = d.get("doc_id", "")
                self.experiment_id     = d.get("experiment_id", "")
                self.status            = d.get("status", "ok")
                self.failed            = d.get("status", "ok") != "ok"
                self.extractions       = d.get("extractions", [])
                self.detected_props    = d.get("detected_props", [])
                self.n_examples_used   = d.get("n_examples_used", 0)
                self.reflection_stats  = d.get("reflection_stats", {})
                self.latency_seconds   = d.get("latency_seconds", 0.0)
                self.n_extractions     = d.get("n_extractions", len(self.extractions))

            def to_result_dict(self):
                return self._d

        proxies = [_Proxy(d) for d in result_dicts if d.get("status", "ok") == "ok"]
        if not proxies:
            logger.warning(f"[{exp_id}] No successful results — skipping metrics.")
            return

        metrics  = aggregate_results(proxies, data)
        base_acc = aggregate_base_accuracy(proxies, data)

        (results_dir / f"{exp_id}_metrics.json").write_text(
            json.dumps(_clean(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
        (results_dir / f"{exp_id}_base_accuracy.json").write_text(
            json.dumps(_clean(base_acc), indent=2, ensure_ascii=False), encoding="utf-8")

        l2 = metrics.get("L2_global", {})
        logger.info(
            f"[{exp_id}] Metrics written — "
            f"Compl={l2.get('extraction_completeness',0):.4f}  "
            f"BaseAcc={base_acc.get('overall',0):.4f}  "
            f"n={len(proxies)}"
        )
    except Exception as exc:
        logger.error(f"[{exp_id}] Could not compute metrics: {exc}")
        logger.info(f"[{exp_id}] Results JSON was saved; re-run the experiment script to recompute metrics.")


def recover_experiment(exp_id: str, results_dir: Path, logs_dir: Path,
                       force: bool = False) -> bool:
    """
    Recover a single experiment.  Returns True if any action was taken.
    """
    results_file = results_dir / f"{exp_id}_results.json"
    existing     = _load_results(results_file)
    have_count   = len(existing)

    # Count docs that are in logs
    log_count = 0
    for fname in (f"{exp_id}_docs.jsonl", f"{exp_id}_full_trace.jsonl"):
        p = logs_dir / fname
        if p.exists():
            log_count = sum(1 for ln in p.open(encoding="utf-8") if ln.strip())
            if log_count:
                break

    if log_count == 0:
        logger.info(f"[{exp_id}] No log files found — nothing to recover.")
        return False

    logger.info(
        f"[{exp_id}] Results JSON has {have_count} doc(s); logs have {log_count} doc(s)."
    )

    if have_count == log_count and not force:
        logger.info(f"[{exp_id}] Counts match — already complete.  Use --force to recompute metrics.")
        return False

    if have_count < log_count or force:
        # First try the incremental checkpoint repair (handles partial writes)
        n_recovered, n_rerun = repair_checkpoint_with_recovery(exp_id, results_dir, logs_dir)

        # Re-check how many we have now
        updated = _load_results(results_file)
        if len(updated) < log_count:
            logger.info(
                f"[{exp_id}] After checkpoint repair: {len(updated)}/{log_count} docs in JSON. "
                f"Rebuilding fully from logs..."
            )
            # Full rebuild from logs (handles corrupt/missing results JSON)
            all_results = rebuild_results_from_logs(exp_id, results_dir, logs_dir)
            if all_results:
                # Merge with anything already in the file
                merged = {r.get("doc_id"): r for r in updated}
                for r in all_results:
                    merged.setdefault(r.get("doc_id", ""), r)
                final_list = list(merged.values())
                results_file.write_text(
                    json.dumps(final_list, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                logger.info(f"[{exp_id}] Wrote {len(final_list)} doc(s) to {results_file.name}")
                updated = final_list

        # Recompute metrics from the recovered results
        compute_and_save_metrics(exp_id, updated, results_dir)
        return True

    return False


def main():
    p = argparse.ArgumentParser(description="Recover experiment results from log files (no API calls).")
    p.add_argument("--exp",     nargs="*", default=None,
                   help="Experiment ID(s) to recover, e.g. B0 B1.  Defaults to B0 B1.")
    p.add_argument("--force",   action="store_true",
                   help="Force metrics recomputation even if result count looks correct.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Override results directory (default: config.get_results_dir())")
    p.add_argument("--logs-dir",    type=Path, default=None,
                   help="Override logs directory (default: config.get_logs_dir())")
    args = p.parse_args()

    results_dir = args.results_dir or config.get_results_dir()
    logs_dir    = args.logs_dir    or config.get_logs_dir()
    exp_ids     = args.exp if args.exp else ["B0", "B1"]

    logger.info(f"Results dir : {results_dir}")
    logger.info(f"Logs dir    : {logs_dir}")

    any_action = False
    for exp_id in exp_ids:
        acted = recover_experiment(exp_id, results_dir, logs_dir, force=args.force)
        any_action = any_action or acted

    if not any_action:
        logger.info("Nothing to recover — all result files appear complete.")
        logger.info("Tip: use --force to force metrics recomputation.")


if __name__ == "__main__":
    main()
