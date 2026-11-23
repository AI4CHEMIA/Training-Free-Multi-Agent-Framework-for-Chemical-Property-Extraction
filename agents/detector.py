"""
agents/detector.py — Agent 1: Property Detector

Scans text and identifies which chemical properties are present (no covariate detection).
"""

import json
import logging

import config
from utils.llm_client import chat_json

logger = logging.getLogger(__name__)

_SYSTEM = """You are a chemical property detection agent for scientific literature.
Scan the passage and identify ALL chemical properties for which a numerical value
is explicitly reported for a specific chemical substance.

For each detected property output:
{
  "property": "<canonical lowercase name>",
  "chemicals_mentioned": ["name1", "name2"]
}

Return ONLY a JSON array. Be exhaustive — include every property with a reported value."""


def _make_oneshot_messages(oneshot_item: dict) -> list[dict]:
    """Format a training pool item as a single detector example message pair."""
    e = oneshot_item.get("extraction", {})
    text = oneshot_item.get("text", "")
    prop = e.get("property_label", e.get("property", "unknown"))
    chem = e.get("chemical_name", "unknown")
    example_output = json.dumps([{"property": prop, "chemicals_mentioned": [chem]}])
    return [
        {"role": "user",      "content": text},
        {"role": "assistant", "content": example_output},
    ]


def detect_properties(text: str, model: str = None, oneshot_item: dict = None) -> list[dict]:
    model    = model or config.MODELS["detector"]
    messages = [{"role": "system", "content": _SYSTEM}]
    if oneshot_item:
        messages.extend(_make_oneshot_messages(oneshot_item))
    messages.append({"role": "user", "content": text})

    result = chat_json(messages, model=model, temperature=0.0)
    if not isinstance(result, list):
        return []
    return [
        {
            "property":            str(r.get("property", "")).strip().lower(),
            "chemicals_mentioned": r.get("chemicals_mentioned", []),
        }
        for r in result if isinstance(r, dict)
    ]
