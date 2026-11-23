#!/usr/bin/env python3
"""experiments/generate_figures.py — Generate all figures from saved results."""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.loader import load_dataset
from data.index import PropertyIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-index", action="store_true")
    args = p.parse_args()

    data        = load_dataset()
    index       = PropertyIndex()
    if args.rebuild_index:
        index.build(data["train_docs"]); index.save()
    else:
        index = PropertyIndex.load_or_build(data["train_docs"])

    results_dir = config.get_results_dir()
    figures_dir = config.get_figures_dir()
    extra = {}
    for fname, key in [("ablation_properties_all.json","ablation"),
                       ("ablation_reflection_metrics.json","reflection_ablation")]:
        path = results_dir / fname
        if path.exists():
            extra[key] = json.loads(path.read_text())

    from figures.visualize import generate_all_figures
    generate_all_figures(results_dir=results_dir, figures_dir=figures_dir,
                         index=index, extra=extra)
    print(f"Figures saved to {figures_dir}/")

if __name__ == "__main__":
    main()
