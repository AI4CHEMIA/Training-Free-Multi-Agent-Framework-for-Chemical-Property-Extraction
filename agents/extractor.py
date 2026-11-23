"""
agents/extractor.py — Agent 2: Extractor
"""

import json
import logging

import config
from utils.llm_client import chat_json

logger = logging.getLogger(__name__)

_SYSTEM = """You are a chemical property information extraction specialist.

EXTRACTION RULES:
1. Extract ALL chemicals with reported property values from the text.
2. Multiple properties for the same chemical → separate entries.
3. chemical_name: if both forms are given in text, then use the SHORT form — formula or abbreviation (e.g. "MgF2", "Gd", "R1234ze(E)"). If only large form is given in text, then only use the large/full form.
   chemical_name_long: full/systematic name if a short form exists (e.g. "magnesium fluoride", "Gadolinium"); otherwise leave "".
4. value: bare number or range ONLY.
   - Single number: "420", "1.37", "5.96e7" Try to keep is as given in the text.
   - Range: "X-Y" format only (e.g. "71-72", "154-303"). Do NOT average ranges.
   - Do NOT include qualifier words ("about", "approximately", "ca.", ">", "<", "~").
   - Do NOT use commas in numbers: "10800" not "10,800". Use decimal point: "2.74" not "2,74".
   - Scientific notation: use the e-notation ("5.96e7" not "5.96 × 10^7").
5. value_approximate: true if the source text used words like "about", "approximately", "ca.", "~", ">", "<", "roughly", etc. Otherwise false.
6. unit: leave "" (empty string) for dimensionless properties (refractive index, dielectric constant, GWP, elongation %, etc.). Never write "dimensionless", "unitless", or "1".
7. Do NOT extract experimental conditions as property values.
8. Do NOT extract properties mentioned without a numerical value.
9. Do NOT hallucinate — only extract what is explicitly in the text.
10. Try to keep units as same as the given text.

OUTPUT SCHEMA (JSON array):
[
  {
    "chemical_name":      "<short form: formula or abbreviation>",
    "chemical_name_long": "<full/systematic name, or empty string>",
    "value":              "<bare number or X-Y range>",
    "value_approximate":  false,
    "unit":               "<unit string, or empty string if dimensionless>",
    "property":           "<lowercase canonical property name>"
  }
]
Return [] if nothing should be extracted."""


def _format_oneshot(oneshot_item: dict) -> str:
    e    = oneshot_item.get("extraction", {})
    text = oneshot_item.get("text", "")
    prop = e.get("property_label", e.get("property", ""))
    entry = json.dumps([{
        "chemical_name": e.get("chemical_name", ""),
        "value":         e.get("value", ""),
        "unit":          e.get("unit", ""),
        "property":      prop,
    }], indent=2)
    return f"### ONE-SHOT EXAMPLE\nTEXT: {text}\nEXTRACTED:\n{entry}\n### END EXAMPLE\n"


def _validate(items: list) -> list[dict]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        e = {
            "chemical_name":      str(item.get("chemical_name", "")).strip(),
            "chemical_name_long": str(item.get("chemical_name_long", "")).strip(),
            "value":              str(item.get("value", "")).strip(),
            "value_approximate":  bool(item.get("value_approximate", False)),
            "unit":               str(item.get("unit", "")).strip(),
            "property":           str(item.get("property", "")).strip().lower(),
        }
        if e["chemical_name"] and e["value"] and e["property"]:
            out.append(e)
    return out


def extract(
    text: str,
    detected_props: list[dict] = None,
    instructions: str = "",
    model: str = None,
    oneshot_item: dict = None,
) -> list[dict]:
    model = model or config.MODELS["extractor"]

    parts = []
    if oneshot_item:
        parts += [_format_oneshot(oneshot_item), ""]
    if instructions:
        parts += [f"### EXTRACTION GUIDANCE\n{instructions}\n", ""]
    if detected_props:
        props = [p["property"] for p in detected_props]
        parts.append(f"TARGET PROPERTIES: {json.dumps(props)}")
        parts.append("")
    parts += ["### TEXT TO EXTRACT FROM", text, "", "Return the JSON array."]

    result = chat_json(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(parts)}],
        model=model, temperature=0.0,
    )
    return _validate(result if isinstance(result, list) else [])
