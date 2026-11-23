"""
utils/recover.py — Recover _results.json from the docs/trace log.

Called automatically by _repair_checkpoint() when --resume detects that docs
are in _completed.txt but missing from _results.json.  This happens when the
run completed all API calls but crashed before (or during) the final JSON write.

Recovery sources (tried in order):
  1. logs/<EXP>_docs.jsonl    — one to_result_dict() record per line (fastest)
  2. logs/<EXP>_full_trace.jsonl — full trace; extractions under 'final_extractions' key

No API calls are made.  Only the serialized outputs already on disk are used.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _trace_to_result_dict(rec: dict) -> dict:
    """Convert a full_trace record to the same shape as to_result_dict()."""
    return {
        "doc_id":           rec.get("doc_id", ""),
        "experiment_id":    rec.get("experiment_id", ""),
        "status":           rec.get("status", "ok"),
        "error":            rec.get("error"),
        "error_kind":       rec.get("error_kind"),
        "extractions":      rec.get("final_extractions", rec.get("extractions", [])),
        "n_extractions":    rec.get("n_extractions", len(rec.get("final_extractions", rec.get("extractions", [])))),
        "detected_props":   rec.get("detected_props", []),
        "n_examples_used":  rec.get("n_examples_used", 0),
        "reflection_stats": rec.get("reflection_stats", {}),
        "latency_seconds":  rec.get("latency_seconds", 0.0),
    }


def recover_results_from_logs(
    results_dir: Path,
    logs_dir: Path,
    exp_id: str,
    orphaned_ids: set[str],
) -> dict[str, dict]:
    """
    Look up orphaned doc_ids in the docs/trace log files.

    Returns {doc_id: result_dict} for every orphaned doc found in the logs.
    Doc IDs not found in any log file are NOT in the returned dict
    (those will need to be re-run).
    """
    results_dir = Path(results_dir)
    logs_dir    = Path(logs_dir)
    recovered: dict[str, dict] = {}

    # ── Source 1: _docs.jsonl (exact to_result_dict() format) ─────────────────
    docs_file = logs_dir / f"{exp_id}_docs.jsonl"
    for rec in _load_jsonl(docs_file):
        did = rec.get("doc_id", "")
        if did in orphaned_ids and did not in recovered:
            recovered[did] = rec

    # ── Source 2: _full_trace.jsonl (full trace, convert on the fly) ───────────
    if len(recovered) < len(orphaned_ids):
        still_missing = orphaned_ids - set(recovered)
        trace_file = logs_dir / f"{exp_id}_full_trace.jsonl"
        for rec in _load_jsonl(trace_file):
            did = rec.get("doc_id", "")
            if did in still_missing:
                recovered[did] = _trace_to_result_dict(rec)

    return recovered


def repair_checkpoint_with_recovery(
    exp_id: str,
    results_dir: Path,
    logs_dir: Path,
) -> tuple[int, int]:
    """
    Cross-check _completed.txt against _results.json.
    For orphaned docs (completed but no result in JSON):
      - Recover from logs if possible → write to _results.json
      - Only un-mark from checkpoint docs that can't be found anywhere

    Returns (n_recovered, n_must_rerun).
    """
    from utils.error_store import ErrorStore

    store    = ErrorStore(results_dir, exp_id)
    done_ids = store.load_completed()
    if not done_ids:
        return 0, 0

    # Load existing results
    results_file = results_dir / f"{exp_id}_results.json"
    existing: list[dict] = []
    if results_file.exists():
        raw = results_file.read_text(encoding="utf-8").strip()
        if raw:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"[Recover] Corrupt {results_file.name} — rebuilding from logs.")

    have_ids  = {r.get("doc_id") for r in existing}
    orphaned  = done_ids - have_ids

    if not orphaned:
        return 0, 0

    logger.info(
        f"[Recover] {exp_id}: {len(orphaned)} doc(s) in checkpoint but missing from results JSON."
    )

    # Try to recover from log files
    recovered = recover_results_from_logs(results_dir, logs_dir, exp_id, orphaned)

    n_recovered = 0
    if recovered:
        for did, result in recovered.items():
            existing.append(result)
            n_recovered += 1
        results_file.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(
            f"[Recover] {exp_id}: recovered {n_recovered}/{len(orphaned)} doc(s) "
            f"from logs → {results_file.name}"
        )

    # Only un-mark docs that truly can't be found anywhere
    truly_missing = orphaned - set(recovered)
    n_must_rerun  = len(truly_missing)
    if truly_missing:
        valid = done_ids - truly_missing
        completed_file = results_dir / f"{exp_id}_completed.txt"
        completed_file.write_text(
            "\n".join(sorted(valid)) + ("\n" if valid else ""),
            encoding="utf-8",
        )
        logger.warning(
            f"[Recover] {exp_id}: {n_must_rerun} doc(s) not found in any log — "
            f"will re-run: {sorted(truly_missing)[:5]}{'...' if n_must_rerun > 5 else ''}"
        )
    else:
        logger.info(f"[Recover] {exp_id}: all orphaned docs recovered — no re-runs needed.")

    return n_recovered, n_must_rerun
