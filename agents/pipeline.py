"""
agents/pipeline.py — OmniPropAgent pipeline (3-agent).

Agents
------
  1. Property Detector  — identifies properties present in the text
  2. Extractor          — produces structured JSON extractions
  3. Reflector          — QA loop; can fix directly or request re-extraction (max 3 iters)

Experiment variants (config.EXPERIMENTS)
-----------------------------------------
  B0  Zero-shot  — no examples in any agent prompt
  B1  1-shot     — each agent prompt contains one training example

File layout per experiment run
-------------------------------
results/<model>/
  <EXP>_results.json          clean extraction outputs only
  <EXP>_metrics.json          aggregated metrics
  <EXP>_base_accuracy.json    base accuracy breakdown
  <EXP>_errors.jsonl          full error record per failed document
  <EXP>_failed_ids.txt        doc_id per line  ← feed to --redo
  <EXP>_completed.txt         checkpoint for --resume
  logs/
    <EXP>_full_trace.jsonl    one line per document: agent decisions + raw LLM calls
    <EXP>_docs.jsonl          alias for backward compat
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from agents.detector  import detect_properties
from agents.extractor import extract
from agents.reflector import reflect_once
from utils.llm_client import (
    LLMAPIError, LLMNetworkError, LLMParseError,
    clear_call_log, get_call_log,
)
from utils.error_store import ErrorStore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  RunResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    doc_id:           str
    text:             str
    experiment_id:    str
    detected_props:   list  = field(default_factory=list)
    n_examples_used:  int   = 0
    extractions:      list  = field(default_factory=list)
    reflection_stats: dict  = field(default_factory=dict)
    latency_seconds:  float = 0.0
    status:           str   = "ok"      # "ok" | "failed" | "partial"
    error:            Optional[str] = None
    error_kind:       Optional[str] = None

    agent_trace: dict = field(default_factory=dict)
    llm_calls:   list = field(default_factory=list)

    def to_result_dict(self) -> dict:
        return {
            "doc_id":           self.doc_id,
            "experiment_id":    self.experiment_id,
            "status":           self.status,
            "error":            self.error,
            "error_kind":       self.error_kind,
            "extractions":      self.extractions,
            "n_extractions":    len(self.extractions),
            "detected_props":   self.detected_props,
            "n_examples_used":  self.n_examples_used,
            "reflection_stats": self.reflection_stats,
            "latency_seconds":  self.latency_seconds,
        }

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def failed(self) -> bool:
        return self.status == "failed"


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _trace(res: RunResult, agent: str, **kwargs):
    res.agent_trace.setdefault(agent, {}).update(kwargs)


def _snapshot_calls(res: RunResult, agent_tag: str):
    calls = get_call_log()
    clear_call_log()
    for c in calls:
        c["pipeline_agent_tag"] = agent_tag
    res.llm_calls.extend(calls)


def _write_full_trace(res: RunResult, logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    full_record = {
        "doc_id":            res.doc_id,
        "experiment_id":     res.experiment_id,
        "status":            res.status,
        "error":             res.error,
        "error_kind":        res.error_kind,
        "source_text":       res.text,
        "final_extractions": res.extractions,
        "n_extractions":     len(res.extractions),
        "agent_trace":       res.agent_trace,
        "llm_calls":         res.llm_calls,
        "n_llm_calls":       len(res.llm_calls),
        "latency_seconds":   res.latency_seconds,
        "reflection_stats":  res.reflection_stats,
        "n_examples_used":   res.n_examples_used,
        "detected_props":    res.detected_props,
        "config_snapshot": {
            "extractor_model":      config.MODELS["extractor"],
            "detector_model":       config.MODELS["detector"],
            "reflector_model":      config.MODELS["reflector"],
            "backend":              config.LLM_BACKEND,
            "max_reflection_iters": config.MAX_REFLECTION_ITERS,
            "use_examples":         config.EXPERIMENTS.get(res.experiment_id, {}).get("use_examples", False),
            "temperature":          config.TEMPERATURE,
        },
    }
    trace_file = logs_dir / f"{res.experiment_id}_full_trace.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_record, ensure_ascii=False, default=str) + "\n")

    docs_file = logs_dir / f"{res.experiment_id}_docs.jsonl"
    with open(docs_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.to_result_dict(), ensure_ascii=False, default=str) + "\n")


def _agent_call(fn, agent_name: str, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (LLMAPIError, LLMNetworkError, LLMParseError):
        raise
    except Exception as exc:
        raise RuntimeError(f"[{agent_name}] Unexpected error: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    text: str,
    doc_id: str,
    experiment_id: str = "B1",
    static_pool: list = None,
    logs_dir: Path = None,
    error_store: Optional[ErrorStore] = None,
    index=None,           # unused; kept for call-site compatibility
) -> RunResult:
    t0  = time.time()
    cfg = config.EXPERIMENTS.get(experiment_id, config.EXPERIMENTS["B1"])
    res = RunResult(doc_id=doc_id, text=text, experiment_id=experiment_id)

    clear_call_log()

    _trace(res, "meta",
           doc_id=doc_id,
           experiment_id=experiment_id,
           extractor_model=config.MODELS["extractor"],
           detector_model=config.MODELS["detector"],
           reflector_model=config.MODELS["reflector"],
           backend=config.LLM_BACKEND,
           timestamp_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    try:
        use_examples = cfg.get("use_examples", False)

        # ── Pick 1-shot example ──────────────────────────────────────────────
        oneshot_item: Optional[dict] = None
        if use_examples and static_pool:
            oneshot_item = random.choice(static_pool)
            res.n_examples_used = 1
            _trace(res, "oneshot",
                   source=oneshot_item.get("source", "?"),
                   property=oneshot_item.get("extraction", {}).get("property_label", "?"))

        # ── Agent 1: Detector ────────────────────────────────────────────────
        t1 = time.time()
        detected = _agent_call(detect_properties, "detector", text,
                               oneshot_item=oneshot_item)
        res.detected_props = detected
        _snapshot_calls(res, "detector")
        _trace(res, "detector",
               detected_properties=detected,
               n_properties_found=len(detected),
               latency=round(time.time() - t1, 3))

        # ── Agent 2: Extractor (initial, fatal on failure) ───────────────────
        t2 = time.time()
        extractions = _agent_call(extract, "extractor",
                                  text, detected,
                                  oneshot_item=oneshot_item)
        _snapshot_calls(res, "extractor")
        _trace(res, "extractor_iter0",
               raw_output=extractions,
               n_extracted=len(extractions),
               latency=round(time.time() - t2, 3))

        res.extractions = extractions
        res.status = "partial"   # extractor succeeded checkpoint

        # ── Agent 3: Reflector loop ──────────────────────────────────────────
        t3 = time.time()
        history = []
        max_iters = config.MAX_REFLECTION_ITERS

        for i in range(max_iters):
            is_final = (i == max_iters - 1)
            try:
                action, new_exts, issues, instructions = reflect_once(
                    text, extractions, detected,
                    oneshot_item=oneshot_item,
                    is_final=is_final,
                )
                _snapshot_calls(res, f"reflector_iter{i+1}")
            except (LLMAPIError, LLMNetworkError, LLMParseError) as exc:
                _snapshot_calls(res, f"reflector_iter{i+1}_failed")
                logger.warning(
                    f"[Pipeline] {doc_id[:50]}: reflection iter {i+1} failed "
                    f"({type(exc).__name__}), keeping current extractions. Error: {exc}"
                )
                history.append({"iteration": i+1, "action": "error",
                                "n_issues": 0, "n_extractions": len(extractions),
                                "error": str(exc)})
                break

            history.append({"iteration": i+1, "action": action,
                            "n_issues": len(issues), "n_extractions": len(new_exts)})
            extractions = new_exts
            _trace(res, f"reflector_iter{i+1}",
                   action=action,
                   n_issues=len(issues),
                   n_extractions=len(extractions),
                   is_final=is_final)

            if action == "pass":
                break

            if action == "re_extract" and not is_final:
                try:
                    t_re = time.time()
                    extractions = _agent_call(extract, "extractor",
                                             text, detected,
                                             instructions=instructions,
                                             oneshot_item=oneshot_item)
                    _snapshot_calls(res, f"extractor_iter{i+1}_redo")
                    _trace(res, f"extractor_iter{i+1}_redo",
                           instructions=instructions,
                           n_extracted=len(extractions),
                           latency=round(time.time() - t_re, 3))
                except (LLMAPIError, LLMNetworkError, LLMParseError) as exc:
                    _snapshot_calls(res, f"extractor_iter{i+1}_redo_failed")
                    logger.warning(
                        f"[Pipeline] {doc_id[:50]}: re-extraction iter {i+1} failed "
                        f"({type(exc).__name__}), keeping current. Error: {exc}"
                    )
                    break
            # action == "fix": extractions already updated above

        res.extractions = extractions
        res.reflection_stats = {
            "iters_used": len(history),
            "converged":  history[-1]["action"] == "pass" if history else True,
            "history":    history,
        }
        _trace(res, "reflector",
               total_latency=round(time.time() - t3, 3),
               iters_used=len(history),
               converged=res.reflection_stats["converged"])

        res.status = "ok"

    # ── Fatal errors ──────────────────────────────────────────────────────────
    except (LLMAPIError, LLMNetworkError, LLMParseError) as exc:
        from utils.error_store import classify_exception
        kind = classify_exception(exc)
        _snapshot_calls(res, "fatal_error")
        logger.error(f"[Pipeline] FATAL {kind} on {doc_id[:60]}: {exc}")
        res.status     = "failed"
        res.error      = str(exc)
        res.error_kind = kind
        _trace(res, "error", kind=kind, message=str(exc))
        if error_store:
            error_store.record_error(doc_id, exc, res.agent_trace, text)

    except Exception as exc:
        _snapshot_calls(res, "fatal_error")
        logger.error(f"[Pipeline] UNEXPECTED on {doc_id[:60]}: {exc}", exc_info=True)
        res.status     = "failed"
        res.error      = str(exc)
        res.error_kind = "unknown_error"
        _trace(res, "error", kind="unknown_error", message=str(exc))
        if error_store:
            error_store.record_error(doc_id, exc, res.agent_trace, text)

    # ── Finalise ──────────────────────────────────────────────────────────────
    res.latency_seconds = round(time.time() - t0, 3)
    _trace(res, "meta",
           timestamp_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           total_latency=res.latency_seconds,
           n_extractions=len(res.extractions),
           n_llm_calls=len(res.llm_calls),
           status=res.status)

    if logs_dir:
        _write_full_trace(res, logs_dir)

    if not res.failed and error_store:
        error_store.record_success(doc_id)

    logger.info(
        f"[Pipeline] {experiment_id} | {res.status.upper():<7} | "
        f"{doc_id[:40]} | {len(res.extractions)} exts | "
        f"{len(res.llm_calls)} LLM calls | {res.latency_seconds:.1f}s"
    )
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    documents: list,
    experiment_id: str,
    index=None,           # unused; kept for call-site compatibility
    train_docs: list = None,
    max_docs: int = None,
    seed: int = config.SEED,
    logs_dir: Path = None,
    results_dir: Path = None,
    skip_completed: bool = False,
) -> tuple[list[RunResult], list[RunResult]]:
    """Returns (good_results, failed_results)."""
    random.seed(seed)
    cfg   = config.EXPERIMENTS.get(experiment_id, {})
    r_dir = Path(results_dir) if results_dir else config.get_results_dir()
    store = ErrorStore(r_dir, experiment_id)
    completed = store.load_completed() if skip_completed else set()

    logger.info(
        f"[Experiment] {experiment_id}: {cfg.get('name','')} | "
        f"{len(documents)} docs | extractor={config.MODELS['extractor']} | "
        f"backend={config.LLM_BACKEND}"
        + (f" | resuming: {len(completed)} already done" if completed else "")
    )

    static_pool: list = []
    if train_docs and cfg.get("use_examples", False):
        for doc in train_docs:
            for e in doc.extractions:
                static_pool.append({
                    "source": "retrieved",
                    "text": doc.text[:400],
                    "extraction": e.to_dict(),
                })

    docs = documents[:max_docs] if max_docs else documents
    good:   list[RunResult] = []
    failed: list[RunResult] = []

    for i, doc in enumerate(docs):
        doc_id = getattr(doc, "text_id", None) or doc.doi
        if doc_id in completed:
            logger.info(f"  Skipping (already completed): {doc_id[:60]}")
            continue

        logger.info(f"  Doc {i+1}/{len(docs)}: {doc_id[:60]}")
        res = run_pipeline(
            text=doc.text,
            doc_id=doc_id,
            experiment_id=experiment_id,
            static_pool=static_pool,
            logs_dir=logs_dir,
            error_store=store,
        )
        (failed if res.failed else good).append(res)

    n_ok, n_failed = len(good), len(failed)
    logger.info(
        f"[Experiment] {experiment_id} done — {n_ok} ok, {n_failed} failed"
        + (f" | error breakdown: {store.summary()['error_kinds']}" if n_failed else "")
    )
    if n_failed:
        logger.info(
            f"  Failed IDs saved to: {r_dir / f'{experiment_id}_failed_ids.txt'}"
        )
        logger.info(f"  Re-run with: python run_all.py --redo {experiment_id}")

    return good, failed
