#!/usr/bin/env python3
"""
experiments/run_ablation_examples.py — A-examples ablation (n=3,5,10,15).

Usage:
  python experiments/run_ablation_examples.py
  python experiments/run_ablation_examples.py --counts 3 5 10
  python experiments/run_ablation_examples.py --max-docs 20
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
    p.add_argument("--counts",   nargs="+", type=int, default=[3, 5, 10, 15])
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    data        = load_dataset()
    index       = PropertyIndex.load_or_build(data["train_docs"])
    results_dir = config.get_results_dir()
    logs_dir    = config.get_logs_dir()
    abl         = {}

    for n in args.counts:
        logger.info(f"\nA-examples: n_examples={n}")
        _orig = config.FINAL_EXAMPLES
        config.FINAL_EXAMPLES = n               # type: ignore
        config.EXPERIMENTS["A3"]["n_examples"] = n

        good, failed = run_experiment(
            documents=data["test_docs"], experiment_id="A3",
            index=index, train_docs=data["train_docs"],
            max_docs=args.max_docs, logs_dir=logs_dir, results_dir=results_dir,
        )
        config.FINAL_EXAMPLES = _orig

        if not good: logger.warning(f"n={n}: no successful results."); continue
        metrics  = aggregate_results(good, data)
        base_acc = aggregate_base_accuracy(good, data)
        abl[str(n)] = {"metrics": metrics, "base_accuracy": base_acc}

        (results_dir / f"ablation_examples_{n}_metrics.json").write_text(
            json.dumps(_clean(metrics), indent=2))
        (results_dir / f"ablation_examples_{n}_base_accuracy.json").write_text(
            json.dumps(_clean(base_acc), indent=2))
        logger.info(f"  n={n} → Compl={metrics.get('L2_global',{}).get('extraction_completeness',0):.4f}  "
                    f"BaseAcc={base_acc['overall']:.4f}  failed={len(failed)}")

    (results_dir / "ablation_examples_all.json").write_text(json.dumps(_clean(abl), indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        ns  = sorted(int(k) for k in abl)
        f1s = [abl[str(n)]["metrics"].get("L2_global",{}).get("extraction_completeness",0) for n in ns]
        bas = [abl[str(n)]["base_accuracy"]["overall"] for n in ns]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(ns, f1s, "o-", color="#E91E63", linewidth=2, markersize=7, label="Macro F1")
        ax.plot(ns, bas, "s--", color="#2196F3", linewidth=2, markersize=7, label="Base Accuracy")
        ax.set_xlabel("# Dynamic Examples"); ax.set_ylabel("Score"); ax.set_ylim(0,1)
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_title("A-examples Ablation")
        out = config.get_figures_dir() / "ablation_examples_curve.png"
        fig.savefig(str(out), bbox_inches="tight"); plt.close()
        logger.info(f"Saved {out}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
