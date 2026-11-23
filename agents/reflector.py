"""
agents/reflector.py — Agent 3: Reflector / QA

Reviews extractions and either passes, fixes them directly, or requests re-extraction
with specific instructions. Runs for at most MAX_REFLECTION_ITERS iterations; on the
final iteration it must produce corrected_extractions regardless.
"""

import json
import logging

import config
from utils.llm_client import chat_json

logger = logging.getLogger(__name__)

_SYSTEM = """You are a quality-assurance agent for chemical property extraction.
Review the extracted data against the source text using this checklist:

1. SCHEMA         — every row has chemical_name, value, unit, property. Unit may be "" for dimensionless.
2. UNIT PLAUSIBLE — units are physically reasonable for the property (e.g. melting point in °C/K, not GPa).
3. COVERAGE       — re-read the text; are any chemical–property pairs missing?
4. NO FALSE POS   — no experimental conditions extracted as property values.
5. NAME FORMAT    — chemical_name must be the SHORT form (formula or abbreviation, e.g. "MgF2" not "magnesium fluoride"). Full name goes in chemical_name_long.
6. CO-EXTRACTION  — if multiple properties appear for the same compound, all are captured.
7. CLEAN VALUES   — value must be a bare number or "X-Y" range. No qualifier words ("about", "~", ">"). No commas in numbers. Decimal point not comma. Dimensionless → unit="". Scientific notation → e-notation ("5.96e7"). If source says "approximately", set value_approximate=true, but strip the word from value.
8. VALUE_APPROX   — value_approximate must be true if source used "about", "approximately", "ca.", "~", ">", "<", "roughly", or similar; false otherwise.
9. PROPERTIES and UNITS     — Try to keep units as same as the given text.

Response (JSON only):
{
  "passes": true/false,
  "action": "pass" | "fix" | "re_extract",
  "issues": [{"type": "<checklist_item>", "description": "...", "correction": "..."}],
  "corrected_extractions": [ ...full corrected list... ] or null,
  "re_extract_instructions": "<specific instructions for the extractor>" or null
}

Action rules:
- "pass"       : passes=true, corrected_extractions=null, re_extract_instructions=null
- "fix"        : passes=false, provide corrected_extractions=[...], re_extract_instructions=null
- "re_extract" : passes=false, corrected_extractions=null, re_extract_instructions="<clear instructions>"

When is_final=true in the prompt: you MUST use action="fix" and provide corrected_extractions."""


def _format_oneshot(oneshot_item: dict) -> str:
    e    = oneshot_item.get("extraction", {})
    text = oneshot_item.get("text", "")
    prop = e.get("property_label", e.get("property", ""))
    extraction = [{"chemical_name": e.get("chemical_name",""), "value": e.get("value",""),
                   "unit": e.get("unit",""), "property": prop}]
    result = {"passes": True, "action": "pass", "issues": [], "corrected_extractions": None,
              "re_extract_instructions": None}
    return (f"### ONE-SHOT EXAMPLE\n"
            f"TEXT: {text}\n"
            f"EXTRACTIONS: {json.dumps(extraction)}\n"
            f"RESULT: {json.dumps(result)}\n"
            f"### END EXAMPLE\n")


def _validate(items) -> list[dict]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict) and item.get("chemical_name") and item.get("value"):
            out.append({
                "chemical_name":      str(item.get("chemical_name", "")).strip(),
                "chemical_name_long": str(item.get("chemical_name_long", "")).strip(),
                "value":              str(item.get("value", "")).strip(),
                "value_approximate":  bool(item.get("value_approximate", False)),
                "unit":               str(item.get("unit", "")).strip(),
                "property":           str(item.get("property", "")).strip().lower(),
            })
    return out


def reflect_once(
    text: str,
    extractions: list[dict],
    detected_props: list[dict] = None,
    model: str = None,
    oneshot_item: dict = None,
    is_final: bool = False,
) -> tuple[str, list[dict], list[dict], str]:
    """
    Returns (action, extractions, issues, re_extract_instructions).
    action is one of: "pass", "fix", "re_extract".
    On is_final=True, action is always "pass" or "fix".
    """
    model = model or config.MODELS["reflector"]

    parts = []
    if oneshot_item:
        parts += [_format_oneshot(oneshot_item), ""]

    prop_ctx = ""
    if detected_props:
        prop_ctx = f"\nExpected properties: {json.dumps([p['property'] for p in detected_props])}"

    final_note = "\n\n[is_final=true] This is your last review. Use action=\"fix\" and provide corrected_extractions." if is_final else ""

    parts.append(
        f"SOURCE TEXT:\n{text}{prop_ctx}\n\n"
        f"CURRENT EXTRACTIONS:\n{json.dumps(extractions, indent=2)}\n"
        f"{final_note}\n"
        "Apply the reflection checklist and return your assessment."
    )

    result = chat_json(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(parts)}],
        model=model, temperature=0.0, max_tokens=3000,
    )

    if not isinstance(result, dict):
        return "pass", extractions, [], ""

    action    = str(result.get("action", "pass")).lower()
    issues    = result.get("issues", []) or []
    corrected = result.get("corrected_extractions")
    instructions = str(result.get("re_extract_instructions") or "").strip()

    # On final iteration, force action to fix
    if is_final and action == "re_extract":
        action = "fix"
        if not corrected:
            corrected = extractions   # keep current as best effort

    if action == "fix" and corrected:
        validated = _validate(corrected)
        logger.info(f"[Reflector] fix: {len(issues)} issue(s), revised to {len(validated)} extractions")
        return "fix", validated, issues, ""

    if action == "re_extract":
        logger.info(f"[Reflector] re_extract: {len(issues)} issue(s), instructions: {instructions[:80]}")
        return "re_extract", extractions, issues, instructions

    # "pass" or unrecognised
    return "pass", extractions, issues, ""
