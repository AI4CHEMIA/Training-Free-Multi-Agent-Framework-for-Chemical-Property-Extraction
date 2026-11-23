"""
config.py — OmniPropAgent: single source of truth for all settings.

Edit this file before running any experiment.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  API KEYS  ←  SET THESE
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY  = ""   # e.g. "sk-ant-..."
OPENAI_API_KEY     = ""   # e.g. "sk-..."
OPENROUTER_API_KEY = ""     # e.g. "sk-or-..."

# Which provider to route LLM calls through:
#   "anthropic"   → https://api.anthropic.com/v1   (OpenAI-compatible)
#   "openai"      → https://api.openai.com/v1
#   "openrouter"  → https://openrouter.ai/api/v1   (any model via OpenRouter)
LLM_BACKEND = "openrouter"

# Optional: site info sent in OpenRouter HTTP headers (recommended by their docs)
OPENROUTER_SITE_URL  = ""   # e.g. "https://yoursite.com"
OPENROUTER_SITE_NAME = ""   # e.g. "OmniPropAgent"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL NAMES  ←  SET THESE
#
#  Each agent uses a different model to avoid self-reinforcing bias.
#
#  Anthropic examples : "claude-opus-4-5", "claude-haiku-4-5-20251001"
#  OpenAI examples    : "gpt-4o", "gpt-4o-mini", "o1-mini"
#  OpenRouter examples: "anthropic/claude-opus-4-5",
#                       "openai/gpt-4o",
#                       "google/gemini-2.5-flash",
#                       "meta-llama/llama-3.3-70b-instruct"
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # Agent 1 – Property Detector
    "detector":  "openai/gpt-5.6-luna-pro",

    # Agent 2 – Extractor (high-volume; cost-efficient)
    #   ↳ This model name (with "/" replaced by "_") becomes the run folder name.
    "extractor": "openai/gpt-5.6-luna-pro",

    # Agent 3 – Reflector / QA (highest quality; catches subtle errors)
    "reflector": "openai/gpt-5.6-luna-pro",

    # Hallucination Judge – Reflector
    "hallucination_judge": "google/gemma-4-31b-it",
}

# ── Embedding model ───────────────────────────────────────────────────────────
# Used to build the FAISS retrieval index.
#
# Options:
#   "text-embedding-3-small"    → OpenAI (requires OPENAI_API_KEY)
#   "text-embedding-3-large"    → OpenAI (requires OPENAI_API_KEY)
#   "openai/text-embedding-3-small" → OpenRouter (requires OPENROUTER_API_KEY)
#   None                        → TF-IDF fallback, no API key needed
#
EMBEDDING_MODEL  = "openai/text-embedding-3-small"
EMBEDDING_DIM    = 1536   # 1536 for text-embedding-3-small; ignored with TF-IDF

# Which API to use for embeddings when EMBEDDING_MODEL is set:
#   "openai"     → uses OPENAI_API_KEY  + api.openai.com
#   "openrouter" → uses OPENROUTER_API_KEY + openrouter.ai
EMBEDDING_BACKEND = "openai"


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MAX_REFLECTION_ITERS = 3     # max reflector iterations per document
MAX_TOKENS           = 4096  # max output tokens per LLM call
TEMPERATURE          = 0.1   # low for deterministic extraction
SEED                 = 42


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
#
#  Results and figures are written to model-specific subdirectories:
#    results/<extractor_model>/
#    figures/<extractor_model>/
#  where <extractor_model> = MODELS["extractor"] with "/" replaced by "_".
# ══════════════════════════════════════════════════════════════════════════════

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"

def _model_slug() -> str:
    return MODELS["extractor"].replace("/", "_")

def get_results_dir() -> Path:
    d = ROOT / "results" / _model_slug()
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_figures_dir() -> Path:
    d = ROOT / "figures" / _model_slug()
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_logs_dir() -> Path:
    d = get_results_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

# Convenience aliases (resolved at import time; re-call the functions if you
# change MODELS["extractor"] at runtime)
# Use config.get_results_dir(), config.get_figures_dir(), config.get_logs_dir()
# everywhere — these are the correct ways to get model-specific output paths.

EXCEL_PATH = Path(
    "./Tasks-for-Multi-Agent-Agentic-AI-for-Universal-Chemical-Material-Property-Extraction-main"
    "/50 Properties Index.xlsx"
)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT  (README §3.2)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_PROPERTY_IDS = frozenset({1, 40, 3, 43, 2, 6, 7, 12, 13, 17, 18, 19, 16, 35, 36, 39, 9})

TEST_TIER1 = frozenset({10, 15, 26, 14, 20, 27, 44})
TEST_TIER2 = frozenset({4, 5, 8, 11, 22, 23, 24, 25, 28, 30, 31, 38, 42, 50, 49})
TEST_TIER3 = frozenset({29, 45, 41, 46, 48, 37, 47, 32, 33, 34, 21})
TEST_PROPERTY_IDS = TEST_TIER1 | TEST_TIER2 | TEST_TIER3

PROPERTY_TIER: dict[int, str] = (
    {p: "train"        for p in TRAIN_PROPERTY_IDS} |
    {p: "tier1_near"   for p in TEST_TIER1}         |
    {p: "tier2_medium" for p in TEST_TIER2}         |
    {p: "tier3_far"    for p in TEST_TIER3}
)

ABLATION_SPLITS: dict[str, frozenset] = {
    "E10": frozenset({1, 40, 3, 43, 2, 6, 12, 13, 17, 16}),
    "E15": TRAIN_PROPERTY_IDS,
    "E20": TRAIN_PROPERTY_IDS | frozenset({10, 15, 26, 14, 20}),
    "E25": TRAIN_PROPERTY_IDS | frozenset({10, 15, 26, 14, 20, 27, 44, 4, 22, 31}),
    "E30": TRAIN_PROPERTY_IDS | frozenset({10, 15, 26, 14, 20, 27, 44, 4, 22, 31, 28, 30, 23, 11, 29}),
}


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT REGISTRY  (README §5.1)
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS: dict[str, dict] = {
    # Both baselines use the full 3-agent pipeline (Detector → Extractor → Reflector).
    # B0: zero-shot — no examples in any agent prompt.
    # B1: 1-shot    — one training example in every agent prompt.
    "B0": {"name": "Zero-shot",
           "use_examples": False},
    "B1": {"name": "1-shot",
           "use_examples": True, "n_examples": 1},
}


# ══════════════════════════════════════════════════════════════════════════════
#  PROPERTY KNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════

PROPERTY_SYNONYMS: dict[str, list[str]] = {
    "melting point":                     ["melting temperature", "tm", "m.p.", "melting temp",
                                          "fusion temperature", "melting point temperature"],
    "boiling point":                     ["boiling temperature", "tb", "b.p.", "boiling temp",
                                          "normal boiling point"],
    "glass transition temperature":      ["tg", "glass transition temp", "glass transition"],
    "critical temperature":              ["tc", "tcr", "critical temp"],
    "critical pressure":                 ["pc", "pcr"],
    "hildebrand solubility parameter":   ["hildebrand parameter", "solubility parameter",
                                          "hildebrand solubility", "hildebrand"],
    "hansen solubility parameter":       ["hansen parameter", "hansen solubility"],
    "log kow":                           ["log pow", "log p", "log kow", "log d",
                                          "log octanol water partition coefficient",
                                          "log octanol/water partition coefficient",
                                          "octanol water partition coefficient",
                                          "partition coefficient"],
    "hardness":                          ["microhardness", "vickers hardness", "hv",
                                          "brinell hardness", "rockwell hardness",
                                          "knoop hardness", "nanoindentation hardness"],
    "specific heat capacity":            ["specific heat", "heat capacity",
                                          "specific heat at constant pressure",
                                          "isobaric heat capacity", "cp"],
    "electrical conductivity":           ["conductivity", "ionic conductivity",
                                          "electronic conductivity"],
    "thermal conductivity":              ["heat conductivity", "thermal conductance"],
    "auto ignition temperature":         ["auto-ignition temperature", "autoignition temperature",
                                          "autoignition temp", "ait", "self-ignition temperature"],
    "flash point":                       ["flash temperature", "fp"],
    "diffusion coefficient":             ["diffusivity", "diffusion constant",
                                          "self-diffusion coefficient", "tracer diffusion coefficient"],
    "molar extinction coefficient":      ["molar absorption coefficient",
                                          "molar absorptivity", "extinction coefficient",
                                          "absorption coefficient"],
    "vapour pressure":                   ["vapor pressure", "vapour pressure", "vp",
                                          "saturated vapour pressure", "saturation vapor pressure"],
}

COVARIATE_MAP: dict[str, list[int]] = {
    "temperature":  [2, 6, 26, 36, 37, 44, 46],
    "subject":      [16, 29, 41, 45, 46],
    "second_phase": [35, 46],
    "wavelength":   [48],
}

UNIT_EQUIVALENCES: list[set[str]] = [
    # Dimensionless
    {"", "dimensionless", "unitless", "1", "—", "-", "n/a", "na", "none"},
    # Temperature
    {"°c", "degrees c", "deg c", "celsius", "° c", "degrees celsius", "℃"},
    {"k", "kelvin"},
    {"°f", "fahrenheit", "degrees f"},
    # Pressure
    {"mpa", "megapascal", "mega pascal"},
    {"gpa", "gigapascal"},
    {"kpa", "kilopascal"},
    {"pa", "pascal"},
    {"bar", "bars"},
    {"mm hg", "mmhg", "torr"},
    # Molar mass / energy per mole
    # (After _nu: "−"→"-", "·"→" ", "⋅"→" ", "^" stripped)
    {"g/mol", "g mol-1", "gmol-1", "g mol -1"},
    {"kj/mol", "kj mol-1", "kjmol-1", "kj mol -1", "kj / mol",
     "kj kcal-1"},                           # kJ/mol variants after _nu normalisation
    {"kcal/mol", "kcal mol-1", "kcalmol-1"},
    {"j/mol", "j mol-1", "j mol -1"},
    {"kj mol-1", "kj mol -1"},               # already covered above; belt-and-suspenders
    # Energy per mass
    {"j/g", "j g-1"},
    {"kj/kg", "kj kg-1"},
    # Specific heat  (after _nu: "·"→" ", no-space variant "j kg-1°c-1" also included)
    {"j/(kg k)", "j kg-1 k-1", "j kg-1 °c-1", "j/(kg °c)", "j kg-1°c-1", "j/(kg·°c)"},
    {"kj/(kg k)", "kj kg-1 k-1", "kj/kg k", "kj kg-1 k -1"},
    # Viscosity
    {"mpa s", "mpas", "mpa.s", "millipascal second", "cp"},
    {"pa s", "pas", "pa.s", "pascal second", "poise"},
    # Density
    {"g/cm3", "g cm-3", "g/cm 3", "kg/l", "mg/ml"},
    {"kg/m3", "kg m-3"},
    # Diffusivity  (cm² s⁻¹ and m² s⁻¹ families; after _nu "·"→" ", "−"→"-", "^-"→"-")
    {"cm2/s", "cm2 s-1", "cm2 s -1", "cm2s-1"},
    {"m2/s", "m2 s-1", "m2 s -1"},
    # Solubility parameter  (MPa^0.5 family; after _nu "^" stripped, "½"→"0.5")
    # "(mpa)(1/2)" results from stripping "^" in "(MPa)^(1/2)"
    {"mpa0.5", "mpa1/2", "(mpa)0.5", "(mpa)1/2", "mpa05",
     "(mpa)^0.5", "(mpa)^1/2", "(mpa)(1/2)", "(mpa)(0.5)", "mpa1/2"},
    # (cal/cm³)^0.5 family (older Hildebrand unit)
    # "(cal/cm3)(1/2)" results from stripping "^" in "(cal/cm3)^(1/2)"
    {"(cal/cm3)0.5", "(cal/cm3)1/2", "(cal/cm 3)0.5", "(cal/cm3)(1/2)", "(cal/cm3)(0.5)"},
    # (cal/m⁻³)^0.5 (yet another older form)
    # "(cal/m3)(1/2)" results from stripping "^" before digits in "(cal/m^3)^(1/2)"
    {"(cal/m-3)0.5", "(cal/m-3)1/2", "(cal/m3)(1/2)", "(cal/m3)0.5", "(cal/m3)1/2"},
    # Thermal conductivity  (after _nu "·"→" ")
    {"w/(m k)", "w/mk", "w m-1 k-1", "w m -1 k -1", "w/m/k", "w m k-1"},
    # Electrical conductivity
    {"s/m", "s m-1", "siemens/m", "siemens per meter", "% iacs"},
    # Molar absorption / extinction coefficient
    {"m-1 cm-1", "mol-1 cm-1", "l mol-1 cm-1", "dm3 mol-1 cm-1",
     "l/(mol cm)", "m -1 cm -1", "mol -1 cm -1", "l mol -1 cm -1",
     "mm-1 cm-1"},
    # Surface tension / spring constant
    {"mn/m", "mn m-1", "n/m", "mn m -1"},
    # Radioactivity  (specific activity units)
    {"curies/g", "curies per g", "ci/g"},
    {"bq/g", "bq g-1"},
    {"gbq/g", "gbq g-1"},
    {"mci/g", "mci g-1"},
    {"bq/kg", "bq kg-1"},
    # Voltage reference electrode
    # Expand set to include bare "v", "v vs rhe", "v vs she", and long-form descriptions
    # so electrochemical potential predictions normalise correctly
    {"v", "vsce", "vsce in nacl solution", "v sce", "v vs sce",
     "v vs rhe", "v vs she", "v vs nhe",
     "v [compared to saturated calomel electrode (sce)",
     "v [compared to saturated calomel electrode",
     "v (vs sce)", "v (vs rhe)", "v (vs she)"},
    # Dielectric strength  (electric field, MV/m family)
    {"mv/m", "mv m-1", "mv m -1", "kv/mm", "kv mm-1"},
    # Molar concentration  (mol/L family for EC50, LC50 measurements)
    {"mol/l", "mol l-1", "mol l -1", "mol dm-3", "mol dm -3", "m"},
    # Micromolar  (μM / uM — common in ecotoxicology)
    {"μm", "um", "μmol/l", "umol/l"},
    # Vickers hardness  (unitless scale; HV, HV0.5, HV1, HV5 are load variants)
    {"hv", "hv0.5", "hv1", "hv5", "hv 0.5", "hv 1", "hv 5", "vhn", "hv0.3"},
    # Dose/toxicity  (body-weight units)
    {"mg/kg", "mg kg-1", "mg/kg body weight", "mg/kg bw", "mg/kg b.w.",
     "mg/kg body weight", "mg kg body weight", "mg g-1"},
    {"µg/l", "ug/l", "µg l-1", "μg/l", "μg l-1"},
    {"mg/l", "mg l-1"},
    # Concentration per volume (μg/mL ≡ μg/L × 1000 — same dim, different scale;
    # kept separate because they are NOT numerically equivalent but are written
    # interchangeably as units in some papers)
    {"μg/ml", "ug/ml", "μg ml-1", "ug ml-1"},
    # Time
    {"h", "hour", "hours", "hr"},
    {"days", "day"},
    {"years", "year"},
]

RESULTS_DIR = get_results_dir()
FIGURES_DIR = get_figures_dir()
LOGS_DIR = get_logs_dir()