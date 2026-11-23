# OmniPropAgent

---

## 1. Setup

```bash
pip install openai faiss-cpu numpy pandas scikit-learn openpyxl matplotlib umap-learn tqdm python-dotenv
```

---

## 2. Configuration

**Everything is configured in a single file: `config.py`.**  
Open it before running anything.


## 3. Run Experiments

### Baselines

Both baselines run the full 3-agent pipeline (Detector → Extractor → Reflector).  
The only difference is whether a training example is included in each agent's prompt.

| ID | Name | Description |
|----|------|-------------|
| B0 | Zero-shot | No examples in any agent prompt — instructions only |
| B1 | 1-shot | One training example injected into every agent prompt |

```bash
python experiments/run_baselines.py              # run B0 and B1
python experiments/run_baselines.py --exp B0     # single experiment
python experiments/run_baselines.py --exp B1
python experiments/run_baselines.py --max-docs 20
python experiments/run_baselines.py --resume     # skip already-completed docs
python experiments/run_baselines.py --redo B1    # re-run failed docs
```

All scripts accept `--max-docs N`, `--redo [EXP_ID]`, and `--resume`.

### Recovery commands

```bash
# Re-run failed documents for B1
python run_all.py --redo B1

# Re-run failed docs for multiple experiments
python run_all.py --redo B0 B1

# Re-run every experiment that has failures
python run_all.py --redo all

# Resume an interrupted run (skip docs already in completed.txt)
python run_all.py --resume

# Resume + redo: resume first, then fix any failures
python run_all.py --resume
python run_all.py --redo B1
```

After a successful redo the recovered results are appended to `_results.json`,  
the doc is removed from `_failed_ids.txt`, metrics are recomputed, and figures  
are regenerated.

---

## 4. Hallucination Analysis

The hallucination judge is a **separate step** run after experiments complete.  
It uses an LLM to classify every unverified prediction (no GT anchor found) as  
`grounded` / `hallucinated` / `ambiguous`.

The judge model and backend are independent from your extractor configuration —  
you can use GPT-4o as judge even if your extractor ran on Claude Haiku.

```bash
# Default judge: config.MODELS["reflector"]
python evaluation/hallucination_check.py --exp B1

# Specify a different judge model
python evaluation/hallucination_check.py --exp B1 --judge-model gpt-4o
python evaluation/hallucination_check.py --exp B1 \
    --judge-model anthropic/claude-opus-4-5 --judge-backend openrouter

# Check multiple experiments
python evaluation/hallucination_check.py --exp B0 B1

# Check every experiment with saved results
python evaluation/hallucination_check.py --exp all

# Override the results directory
python evaluation/hallucination_check.py --exp B1 --results-dir /path/to/results

# Control batch size (claims per judge call)
python evaluation/hallucination_check.py --exp B1 --batch-size 10
```
