"""
utils/error_store.py — Persistent error and checkpoint tracking.

Writes three sidecar files alongside each experiment's results:

  <EXP>_errors.jsonl    — one record per failed document (full trace + error)
  <EXP>_failed_ids.txt  — one doc_id per line (for --redo)
  <EXP>_completed.txt   — one doc_id per line (checkpoint; skip on resume)

The main results file (<EXP>_results.json) never receives a faulty record.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Error classification ───────────────────────────────────────────────────────

class ErrorKind:
    API        = "api_error"          # HTTP / auth / rate-limit from LLM provider
    NETWORK    = "network_error"      # connection timeout, DNS failure
    PARSE      = "parse_error"        # JSON decode failed, schema mismatch
    EMPTY      = "empty_output"       # LLM returned empty / whitespace string
    VALIDATION = "validation_error"   # extraction result failed schema check
    UNKNOWN    = "unknown_error"      # anything else


def classify_exception(exc: Exception) -> str:
    """Map an exception to one of the ErrorKind strings."""
    name = type(exc).__name__.lower()
    msg  = str(exc).lower()

    # OpenAI SDK surface
    if any(k in name for k in ("apierror", "apiconnectionerror", "autherror",
                                "ratelimiterror", "apistatus")):
        return ErrorKind.API
    if any(k in name for k in ("timeout", "connectionerror", "connect",
                                "gaierror", "sslerror")):
        return ErrorKind.NETWORK
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return ErrorKind.NETWORK
    if any(k in name for k in ("jsondecode", "valueerror")) and "json" in msg:
        return ErrorKind.PARSE
    if "parse" in msg or "decode" in msg:
        return ErrorKind.PARSE
    return ErrorKind.UNKNOWN


# ── Store class ───────────────────────────────────────────────────────────────

class ErrorStore:
    """
    Manages error/checkpoint files for one experiment run.

    Usage
    -----
    store = ErrorStore(results_dir, experiment_id)
    store.record_error(doc_id, exc, partial_trace)
    store.record_success(doc_id)
    store.load_completed()   → set of already-done doc_ids
    store.load_failed()      → list of failed doc_ids
    """

    def __init__(self, results_dir: Path, experiment_id: str):
        self.results_dir    = Path(results_dir)
        self.experiment_id  = experiment_id
        self._errors_file   = self.results_dir / f"{experiment_id}_errors.jsonl"
        self._failed_file   = self.results_dir / f"{experiment_id}_failed_ids.txt"
        self._completed_file = self.results_dir / f"{experiment_id}_completed.txt"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_error(
        self,
        doc_id:       str,
        exc:          Exception,
        agent_trace:  dict = None,
        partial_text: str  = "",
    ):
        """
        Append a failed document to the error log and failed-ids list.
        The main results file is NOT touched.
        """
        kind = classify_exception(exc)
        record = {
            "doc_id":       doc_id,
            "experiment_id": self.experiment_id,
            "error_kind":   kind,
            "error_type":   type(exc).__name__,
            "error_message": str(exc),
            "traceback":    traceback.format_exc(),
            "agent_trace":  agent_trace or {},
            "text_snippet": partial_text[:200] if partial_text else "",
        }

        # errors.jsonl
        with open(self._errors_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        # failed_ids.txt  (skip duplicates)
        existing = self.load_failed_set()
        if doc_id not in existing:
            with open(self._failed_file, "a", encoding="utf-8") as f:
                f.write(doc_id + "\n")

        logger.warning(f"[ErrorStore] {self.experiment_id} | {kind} | {doc_id[:60]}: {exc}")

    def record_success(self, doc_id: str):
        """Mark a document as successfully completed (checkpoint)."""
        with open(self._completed_file, "a", encoding="utf-8") as f:
            f.write(doc_id + "\n")

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_completed(self) -> set[str]:
        """Return set of doc_ids already successfully processed."""
        if not self._completed_file.exists():
            return set()
        return {l.strip() for l in self._completed_file.read_text().splitlines() if l.strip()}

    def load_failed_set(self) -> set[str]:
        if not self._failed_file.exists():
            return set()
        return {l.strip() for l in self._failed_file.read_text().splitlines() if l.strip()}

    def load_failed(self) -> list[str]:
        """Return ordered list of failed doc_ids (for --redo)."""
        if not self._failed_file.exists():
            return []
        return [l.strip() for l in self._failed_file.read_text().splitlines() if l.strip()]

    def load_error_records(self) -> list[dict]:
        """Return all error records from the JSONL file."""
        if not self._errors_file.exists():
            return []
        records = []
        for line in self._errors_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    # ── Redo helpers ──────────────────────────────────────────────────────────

    def clear_failed(self, doc_id: str):
        """Remove a doc_id from the failed list (after successful redo)."""
        ids = self.load_failed()
        ids = [i for i in ids if i != doc_id]
        self._failed_file.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    def summary(self) -> dict:
        return {
            "n_completed": len(self.load_completed()),
            "n_failed":    len(self.load_failed()),
            "error_kinds": _count_kinds(self.load_error_records()),
        }


def _count_kinds(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        k = r.get("error_kind", "unknown")
        counts[k] = counts.get(k, 0) + 1
    return counts


# ── Convenience: merge redo results into existing results file ────────────────

def merge_redo_results(
    results_file: Path,
    redo_results: list,
    store: ErrorStore,
):
    """
    Append successfully-redone results into the main results JSON file
    and remove them from the failed list.
    """
    if not redo_results:
        return

    existing = []
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_ids = {r.get("doc_id") for r in existing}
    added = 0
    for rr in redo_results:
        if hasattr(rr, "to_result_dict"):
            d = rr.to_result_dict()
        elif hasattr(rr, "to_dict"):
            d = rr.to_dict()
        else:
            d = rr
        if d.get("doc_id") not in existing_ids:
            existing.append(d)
            store.clear_failed(d["doc_id"])
            added += 1

    results_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"[ErrorStore] Merged {added} redo result(s) → {results_file.name}")
