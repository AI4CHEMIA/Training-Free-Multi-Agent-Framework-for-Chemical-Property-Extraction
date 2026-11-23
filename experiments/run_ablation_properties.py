#!/usr/bin/env python3
"""
experiments/run_ablation_properties.py — A-properties ablation (E10→E30).

Usage:
  python experiments/run_ablation_properties.py
  python experiments/run_ablation_properties.py --splits E10 E15
  python experiments/run_ablation_properties.py --max-docs 20
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.loader import load_dataset
from data.index import PropertyIndex
from agents.pipeline import run_experiment
from evaluation.metrics import aggregate_results, aggregate_base_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(i) for i in o]
    if isinstance(o, float) and (o != o or o == float("inf") or o == float("-inf")): return None
    return o

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits",   nargs="+", default=sorted(config.ABLATION_SPLITS))
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    data        = load_dataset()
    results_dir = config.get_results_dir()
    logs_dir    = config.get_logs_dir()
    all_metrics = {}

    for split_id in args.splits:
        train_ids = config.ABLATION_SPLITS.get(split_id)
        if not train_ids: logger.warning(f"Unknown split: {split_id}"); continue

        logger.info(f"\n[Ablation] {split_id}: {len(train_ids)} training properties")
        train_subset = [d for d in data["train_docs"]
                        if any(pid in train_ids for pid in d.prop_ids)]
        sub_index = PropertyIndex(); sub_index.build(train_subset)

        _orig = config.TRAIN_PROPERTY_IDS
        config.TRAIN_PROPERTY_IDS = train_ids       # type: ignore
        good, failed = run_experiment(
            documents=data["test_docs"], experiment_id="A3",
            index=sub_index, train_docs=train_subset,
            max_docs=args.max_docs, logs_dir=logs_dir, results_dir=results_dir,
        )
        config.TRAIN_PROPERTY_IDS = _orig

        if not good: logger.warning(f"[{split_id}] No successful results."); continue
        metrics  = aggregate_results(good, data)
        base_acc = aggregate_base_accuracy(good, data)
        all_metrics[split_id] = metrics

        (results_dir / f"ablation_{split_id}_metrics.json").write_text(
            json.dumps(_clean(metrics), indent=2))
        (results_dir / f"ablation_{split_id}_base_accuracy.json").write_text(
            json.dumps(_clean(base_acc), indent=2))
        logger.info(f"  Compl={metrics.get('L2_global',{}).get('extraction_completeness',0):.4f}  "
                    f"BaseAcc={base_acc['overall']:.4f}  failed={len(failed)}")

    (results_dir / "ablation_properties_all.json").write_text(
        json.dumps(_clean(all_metrics), indent=2))
    try:
        from figures.visualize import fig3_saturation_curve
        fig3_saturation_curve(all_metrics, figures_dir=config.get_figures_dir())
    except Exception as e:
        logger.warning(f"Figure 3 failed: {e}")

if __name__ == "__main__":
    main()
