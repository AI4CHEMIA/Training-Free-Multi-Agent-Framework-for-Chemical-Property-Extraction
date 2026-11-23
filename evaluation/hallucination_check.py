"""
evaluation/hallucination_check.py — Standalone hallucination analysis.

Reads saved results files, runs LLM-as-judge on every unverified prediction,
and writes per-document verdicts + aggregate summary.

Usage
-----
  # Judge model from config.MODELS["hallucination_judge"]
  python evaluation/hallucination_check.py --exp A3

  # Specify a different judge model
  python evaluation/hallucination_check.py --exp A3 --judge-model gpt-4o
  python evaluation/hallucination_check.py --exp A3 --judge-model anthropic/claude-opus-4-5

  # Run for multiple experiments in the current model folder
  python evaluation/hallucination_check.py --exp A3 B0 B5

  # Run for all experiments in the current model folder
  python evaluation/hallucination_check.py --exp all

  # Run for ALL models in results/ and ALL experiments in each — one by one
  python evaluation/hallucination_check.py --all-models

  # Same but override the judge model
  python evaluation/hallucination_check.py --all-models --judge-model google/gemini-2.5-flash

  # Change the backend used for the judge (independently of LLM_BACKEND)
  python evaluation/hallucination_check.py --exp A3 \\
      --judge-model openai/gpt-4o --judge-backend openrouter

Output files (in results/<extractor_model>/)
  <EXP>_hallucination_check.json    — per-document verdicts + counts
  <EXP>_hallucination_summary.json  — aggregate statistics only
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.loader import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are a scientific fact-checking agent specialised in chemistry.

You are given a source text passage and a list of chemical property claims that
were extracted by an AI system but could NOT be verified against a reference
database (they may be correct extractions the database missed, or they may be
hallucinated).

For EACH claim, determine:
  - "grounded"      : the value CAN be traced to the source text for that compound
  - "hallucinated"  : the value contradicts the text, belongs to a different compound,
                      or is simply not present in the text
  - "ambiguous"     : the text mentions the compound and property but the exact value
                      is unclear (e.g. given as a range, comparison, or approximation
                      that does not clearly match the claim)

Rules:
  1. Only accept "grounded" if the claim's compound, property, and value are all
     explicitly traceable to the source text.
  2. Legitimate unit conversions (e.g. K to °C) still count as grounded if the
     numeric relationship is correct.
  3. Do NOT penalise correct extractions simply because the database missed them.
  4. If the source text is truncated and the relevant portion is missing, mark "ambiguous".

Return a JSON array, one object per claim (in the same order):
[
  {
    "claim_index":  <int>,
    "verdict":      "grounded" | "hallucinated" | "ambiguous",
    "confidence":   "high" | "medium" | "low",
    "evidence":     "<exact quote or phrase from source text supporting this verdict, or empty string>",
    "reason":       "<one sentence explaining the verdict>"
  }
]
"""


def _build_judge_client(judge_backend: str, judge_model: str):
    """Build an OpenAI-compatible client for the judge (may differ from main backend)."""
    from openai import OpenAI
    import os

    backend = (judge_backend or config.LLM_BACKEND).lower()
    if backend == "anthropic":
        api_key = config.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        return OpenAI(api_key=api_key,
                      base_url="https://api.anthropic.com/v1/",
                      default_headers={"anthropic-version": "2023-06-01"})
    elif backend == "openrouter":
        api_key = config.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        extra = {"HTTP-Referer": config.OPENROUTER_SITE_URL,
                 "X-Title":      config.OPENROUTER_SITE_NAME}
        return OpenAI(api_key=api_key,
                      base_url="https://openrouter.ai/api/v1",
                      default_headers={k: v for k, v in extra.items() if v})
    else:
        api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        return OpenAI(api_key=api_key)


def _call_judge(client, judge_model: str, source_text: str, claims: list[dict]) -> list[dict]:
    """Single judge API call. Returns list of verdict dicts."""
    import re, time

    claims_json = json.dumps([
        {
            "claim_index":   i,
            "chemical_name": c.get("chemical_name", ""),
            "property":      c.get("property", ""),
            "value":         c.get("value", ""),
            "unit":          c.get("unit", ""),
            "covariate":     c.get("covariate_value"),
        }
        for i, c in enumerate(claims)
    ], indent=2)

    user_msg = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"CLAIMS TO VERIFY ({len(claims)}):\n{claims_json}\n\n"
        "Return the JSON verdict array."
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=3000,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            if not raw:
                finish = getattr(choice, "finish_reason", "unknown")
                # OpenRouter may surface errors in a provider-specific field
                native = getattr(resp, "model_extra", {}) or {}
                logger.warning(
                    f"Judge returned empty content (attempt {attempt+1}). "
                    f"finish_reason={finish!r}  provider_info={native.get('provider',{})}"
                )
                # Treat empty-content as a retryable error
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw).strip()
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Judge JSON parse error (attempt {attempt+1}): {exc}. "
                f"Raw (first 300 chars): {raw[:300]!r}"
            )
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(f"Judge API error (attempt {attempt+1}): {exc}. "
                           + (f"Retrying in {wait}s…" if attempt < 2 else "Giving up."))
            if attempt < 2:
                time.sleep(wait)
    return []


# ── Identify unverified predictions ──────────────────────────────────────────

def _find_unverified(preds: list[dict], gts: list[dict]) -> list[dict]:
    """Return predictions that have no matching anchor in the GT list."""
    from evaluation.metrics import name_match, property_match
    return [
        p for p in preds
        if not any(
            name_match(p.get("chemical_name",""), g.get("chemical_name",""))
            and property_match(p.get("property",""), g.get("property_label",""))
            for g in gts
        )
    ]


# ── Per-experiment runner ─────────────────────────────────────────────────────

def run_hallucination_check(
    exp_id:        str,
    results_dir:   Path,
    dataset:       dict,
    judge_model:   str  = None,
    judge_backend: str  = None,
    batch_size:    int  = 15,
) -> dict:
    """
    Run hallucination judge for one experiment.

    Parameters
    ----------
    exp_id         : e.g. "A3"
    results_dir    : model-specific results directory
    dataset        : loaded dataset dict
    judge_model    : model to use as judge (default: config.MODELS["reflector"])
    judge_backend  : backend for judge; None = use config.LLM_BACKEND
    batch_size     : max claims per single judge API call

    Output files
    ------------
    <EXP>_hallucination_check.json   — full per-document verdicts
    <EXP>_hallucination_summary.json — aggregate only (no per-doc detail)
    """
    judge_model = judge_model or config.MODELS.get("hallucination_judge", "google/gemini-2.5-flash-lite")
    results_file = results_dir / f"{exp_id}_results.json"

    if not results_file.exists():
        logger.warning(f"[{exp_id}] No results file found: {results_file}")
        return {}

    records: list[dict] = json.loads(results_file.read_text(encoding="utf-8"))
    if not records:
        logger.info(f"[{exp_id}] Empty results file.")
        return {}

    # Build GT and text lookups keyed by text_id ("{doi}#{example_id}"),
    # which matches the doc_id field written to *_results.json.
    gt_by_doi:   dict[str, list] = defaultdict(list)
    text_by_doi: dict[str, str]  = {}
    for doc in dataset.get("documents", []):
        key = doc.text_id or doc.doi   # text_id preferred; bare doi as fallback
        for e in doc.extractions:
            gt_by_doi[key].append(e.to_dict())
        text_by_doi[key] = doc.text

    logger.info(
        f"[{exp_id}] Hallucination check | "
        f"judge={judge_model} | backend={judge_backend or config.LLM_BACKEND} | "
        f"{len(records)} documents"
    )

    client = _build_judge_client(judge_backend, judge_model)

    doc_reports      = []
    total_pred       = 0
    total_novel      = 0
    total_grounded   = 0
    total_halluc     = 0
    total_ambiguous  = 0

    for rec_idx, record in enumerate(records):
        doi   = record.get("doc_id", "")
        preds = record.get("extractions", [])
        gts   = gt_by_doi.get(doi, [])
        text  = text_by_doi.get(doi, "")

        total_pred  += len(preds)
        unverified   = _find_unverified(preds, gts)
        total_novel += len(unverified)

        if not unverified:
            doc_reports.append({
                "doc_id":            doi,
                "n_predictions":     len(preds),
                "n_unverified":      0,
                "n_grounded":        0,
                "n_hallucinated":    0,
                "n_ambiguous":       0,
                "halluc_rate":       0.0,
                "verdicts":          [],
            })
            continue

        # Run judge in batches
        all_verdicts: list[dict] = []
        for start in range(0, len(unverified), batch_size):
            batch    = unverified[start : start + batch_size]
            verdicts = _call_judge(client, judge_model, text, batch)

            # Merge: attach original claim data to each verdict
            verdict_map = {v.get("claim_index", i): v
                           for i, v in enumerate(verdicts)
                           if isinstance(v, dict)}
            for local_i, claim in enumerate(batch):
                global_i = start + local_i
                v = verdict_map.get(local_i, {})
                all_verdicts.append({
                    "claim_index":   global_i,
                    "chemical_name": claim.get("chemical_name", ""),
                    "property":      claim.get("property", ""),
                    "value":         claim.get("value", ""),
                    "unit":          claim.get("unit", ""),
                    "covariate":     claim.get("covariate_value"),
                    "verdict":       v.get("verdict", "ambiguous"),
                    "confidence":    v.get("confidence", "low"),
                    "evidence":      v.get("evidence", ""),
                    "reason":        v.get("reason", "judge unavailable"),
                })

        n_grounded  = sum(1 for v in all_verdicts if v["verdict"] == "grounded")
        n_halluc    = sum(1 for v in all_verdicts if v["verdict"] == "hallucinated")
        n_ambiguous = sum(1 for v in all_verdicts if v["verdict"] == "ambiguous")
        total_grounded  += n_grounded
        total_halluc    += n_halluc
        total_ambiguous += n_ambiguous

        doc_reports.append({
            "doc_id":            doi,
            "n_predictions":     len(preds),
            "n_unverified":      len(unverified),
            "n_grounded":        n_grounded,
            "n_hallucinated":    n_halluc,
            "n_ambiguous":       n_ambiguous,
            "halluc_rate":       round(n_halluc  / max(len(unverified), 1), 4),
            "grounded_rate":     round(n_grounded / max(len(unverified), 1), 4),
            "verdicts":          all_verdicts,
        })

        if (rec_idx + 1) % 10 == 0 or (rec_idx + 1) == len(records):
            logger.info(
                f"  [{exp_id}] {rec_idx+1}/{len(records)} docs | "
                f"novel={total_novel} grounded={total_grounded} "
                f"halluc={total_halluc} ambiguous={total_ambiguous}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "experiment_id":              exp_id,
        "extractor_model":            config.MODELS["extractor"],
        "judge_model":                judge_model,
        "judge_backend":              judge_backend or config.LLM_BACKEND,
        "n_documents":                len(records),
        "total_predictions":          total_pred,
        "total_novel_claims":         total_novel,
        "total_grounded":             total_grounded,
        "total_hallucinated":         total_halluc,
        "total_ambiguous":            total_ambiguous,
        # Rates relative to novel claims
        "grounded_rate_of_novel":     round(total_grounded  / max(total_novel, 1), 4),
        "halluc_rate_of_novel":       round(total_halluc    / max(total_novel, 1), 4),
        "ambiguous_rate_of_novel":    round(total_ambiguous / max(total_novel, 1), 4),
        # Rates relative to ALL predictions
        "novel_rate_of_total":        round(total_novel   / max(total_pred, 1), 4),
        "halluc_rate_of_total":       round(total_halluc  / max(total_pred, 1), 4),
        "grounded_rate_of_total":     round(total_grounded/ max(total_pred, 1), 4),
    }

    # Full report (includes per-doc verdicts)
    full_report = {**summary, "per_document": doc_reports}
    (results_dir / f"{exp_id}_hallucination_check.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # Summary only (for quick inspection / figures)
    (results_dir / f"{exp_id}_hallucination_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    logger.info(
        f"[{exp_id}] Done. novel={total_novel} grounded={total_grounded} "
        f"halluc={total_halluc} "
        f"(halluc_rate_of_novel={summary['halluc_rate_of_novel']:.3f} | "
        f"halluc_rate_of_total={summary['halluc_rate_of_total']:.3f})"
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_table(summaries: dict, model_label: str = ""):
    """Print a summary table.

    summaries: {exp_id: summary_dict} for a single model folder.
    """
    w = 95
    label = f"  [{model_label}]" if model_label else ""
    print(f"\n{'='*w}{label}")
    print(f"{'Exp':<6} {'Docs':>5} {'Pred':>7} {'Novel':>7} {'Grnded':>8} "
          f"{'Halluc':>8} {'Ambig':>7} {'Hal%(nov)':>10} {'Hal%(all)':>10}")
    print(f"{'─'*w}")
    for exp_id, s in summaries.items():
        if not s:
            continue
        print(
            f"{exp_id:<6} "
            f"{s.get('n_documents', 0):>5} "
            f"{s.get('total_predictions', 0):>7} "
            f"{s['total_novel_claims']:>7} "
            f"{s['total_grounded']:>8} "
            f"{s['total_hallucinated']:>8} "
            f"{s['total_ambiguous']:>7} "
            f"{s['halluc_rate_of_novel']*100:>9.1f}% "
            f"{s['halluc_rate_of_total']*100:>9.1f}%"
        )
    print(f"{'='*w}\n")


def _discover_model_dirs(results_root: Path) -> list[Path]:
    """Return all subdirectories of results_root that contain at least one *_results.json."""
    return sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and any(d.glob("*_results.json"))
    )


def _exp_ids_in_dir(model_dir: Path) -> list[str]:
    """Return sorted experiment IDs found in model_dir."""
    return sorted(set(
        p.stem.split("_")[0]
        for p in model_dir.glob("*_results.json")
    ))


def main():
    p = argparse.ArgumentParser(
        description="Standalone hallucination analysis for OmniPropAgent results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--exp", nargs="+", default=["all"], metavar="EXP_ID",
        help="Experiment IDs to check, or 'all' for every saved experiment. "
             "Ignored when --all-models is set.",
    )
    p.add_argument(
        "--all-models", action="store_true",
        help="Discover all model folders under results/ and run every experiment in each.",
    )
    p.add_argument(
        "--judge-model", default=None, metavar="MODEL",
        help="Model to use as judge (default: config.MODELS['hallucination_judge']).",
    )
    p.add_argument(
        "--judge-backend", default=None, metavar="BACKEND",
        help="Backend for judge: anthropic | openai | openrouter "
             "(default: same as config.LLM_BACKEND).",
    )
    p.add_argument(
        "--batch-size", type=int, default=15,
        help="Max unverified claims per judge API call (default: 15).",
    )
    p.add_argument(
        "--results-dir", default=None, metavar="PATH",
        help="Override the model results directory (default: model-specific dir from "
             "config). Ignored when --all-models is set.",
    )
    args = p.parse_args()

    data = load_dataset()

    if args.all_models:
        # ── Multi-model mode: iterate every model folder in results/ ──────────
        results_root = config.ROOT / "results"
        model_dirs   = _discover_model_dirs(results_root)
        if not model_dirs:
            print(f"No model result folders found under {results_root}")
            return

        logger.info(f"Found {len(model_dirs)} model folder(s): "
                    + ", ".join(d.name for d in model_dirs))

        all_summaries: dict[str, dict[str, dict]] = {}  # {model_name: {exp_id: summary}}
        for model_dir in model_dirs:
            exp_ids = _exp_ids_in_dir(model_dir)
            if not exp_ids:
                logger.info(f"  Skipping {model_dir.name} — no *_results.json found")
                continue

            logger.info(f"\n── Model: {model_dir.name} | experiments: {exp_ids} ──")
            summaries: dict[str, dict] = {}
            for exp_id in exp_ids:
                summaries[exp_id] = run_hallucination_check(
                    exp_id        = exp_id,
                    results_dir   = model_dir,
                    dataset       = data,
                    judge_model   = args.judge_model,
                    judge_backend = args.judge_backend,
                    batch_size    = args.batch_size,
                )
            _print_table(summaries, model_label=model_dir.name)
            all_summaries[model_dir.name] = summaries

        print(f"All results saved to sub-directories of: {results_root}/")

    else:
        # ── Single-model mode (original behaviour) ────────────────────────────
        results_dir = Path(args.results_dir) if args.results_dir else config.get_results_dir()

        # Resolve experiment IDs
        if args.exp == ["all"]:
            exp_ids = _exp_ids_in_dir(results_dir)
            if not exp_ids:
                print(f"No results files found in {results_dir}")
                return
        else:
            exp_ids = args.exp

        summaries: dict[str, dict] = {}
        for exp_id in exp_ids:
            summaries[exp_id] = run_hallucination_check(
                exp_id        = exp_id,
                results_dir   = results_dir,
                dataset       = data,
                judge_model   = args.judge_model,
                judge_backend = args.judge_backend,
                batch_size    = args.batch_size,
            )

        _print_table(summaries, model_label=results_dir.name)
        print(f"Full results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
