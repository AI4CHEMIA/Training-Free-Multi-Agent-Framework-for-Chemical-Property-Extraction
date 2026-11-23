#!/usr/bin/env python3
"""
experiments/run_ablation_reflection.py — A-reflection ablation. Produces Fig 8.

Usage:
  python experiments/run_ablation_reflection.py
  python experiments/run_ablation_reflection.py --max-docs 30
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.loader import load_dataset
from data.index import PropertyIndex
from agents.pipeline import RunResult
from evaluation.metrics import aggregate_results, aggregate_base_accuracy
from utils.llm_client import LLMAPIError, LLMNetworkError, LLMParseError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    from agents.detector import detect_properties
    from agents.generator import build_dynamic_examples, format_examples_for_prompt
    from agents.extractor import extract
    from agents.reflector import reflect_once

    data        = load_dataset()
    index       = PropertyIndex.load_or_build(data["train_docs"])
    results_dir = config.get_results_dir()
    docs        = data["test_docs"][:args.max_docs] if args.max_docs else data["test_docs"]

    iter_results: dict[int, list] = {i: [] for i in range(4)}
    for doc in docs:
        try:
            detected = detect_properties(doc.text)
            selected = build_dynamic_examples(doc.text, detected, index)
            ex_str   = format_examples_for_prompt(selected)
            exts     = extract(doc.text, detected, ex_str)
            rr0 = RunResult(doc_id=doc.doi, text=doc.text, experiment_id="refl_iter0")
            rr0.extractions = list(exts); rr0.status = "ok"
            iter_results[0].append(rr0)
            current = list(exts)
            for i in range(1, 4):
                try:
                    passed, current, _ = reflect_once(doc.text, current, detected)
                except (LLMAPIError, LLMNetworkError, LLMParseError) as exc:
                    logger.warning(f"Reflection iter {i} failed for {doc.doi}: {exc}")
                    break
                rr = RunResult(doc_id=doc.doi, text=doc.text, experiment_id=f"refl_iter{i}")
                rr.extractions = list(current); rr.status = "ok"
                iter_results[i].append(rr)
                if passed:
                    for j in range(i+1, 4):
                        rc = RunResult(doc_id=doc.doi, text=doc.text, experiment_id=f"refl_iter{j}")
                        rc.extractions = list(current); rc.status = "ok"
                        iter_results[j].append(rc)
                    break
        except Exception as e:
            logger.error(f"Error on {doc.doi}: {e}")

    per_iter = {}
    for i, rs in iter_results.items():
        if rs:
            m  = aggregate_results(rs, data)
            ba = aggregate_base_accuracy(rs, data)
            per_iter[str(i)] = {"extraction_completeness": m.get("L2_global",{}).get("extraction_completeness",0),
                                 "base_accuracy": ba["overall"]}
            logger.info(f"  Iter {i} → Compl={m.get('L2_global',{}).get('extraction_completeness',0):.4f}  "
                        f"BaseAcc={ba['overall']:.4f}")

    (results_dir / "ablation_reflection_metrics.json").write_text(json.dumps(per_iter, indent=2), encoding=utf-8)
    try:
        from figures.visualize import fig8_reflection_waterfall
        fig8_reflection_waterfall(per_iter, figures_dir=config.get_figures_dir())
    except Exception as e:
        logger.warning(f"Figure 8 failed: {e}")

if __name__ == "__main__":
    main()
