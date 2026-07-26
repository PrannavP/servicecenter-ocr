"""
AI semantic field-mapper for the "Snap-to-Job-Card" flow.

The rigid FIELD_MAP in extracted_key_value.py only fills a field when a form
label matches a hard-coded string exactly. Real job-card sheets vary ("Reg No",
"Vehicle No.", "Plate", handwriting quirks), so many fields come back blank.

This module hands the raw Textract key-values + OCR lines to an LLM and asks it
to map them onto the canonical job-card schema, normalise the values, and report
a confidence per field. It degrades gracefully:

    AWS Bedrock (Claude)  ->  Anthropic API  ->  rule-based fallback

so /upload_v3 always returns a usable result, with or without model access.
"""

import os
import json
import re
import urllib.request

SCHEMA_FIELDS = [
    "customer_name",
    "customer_address",
    "customer_number",
    "vehicle_type",
    "vehicle_chasis_number",
    "vehicle_registration_number",
    "date_time",
    "current_odometer_reading",
    "fuel_level",
    "service_type",
    "expected_delivery_date_time",
    "service_advisor",
    "helmet_kept",
]

_INSTRUCTIONS = (
    "You map noisy OCR output from a vehicle service-center job-card sheet onto a "
    "fixed JSON schema. You are given the key/value pairs AWS Textract found and the "
    "raw text lines.\n\n"
    "Rules:\n"
    "- Map values onto these keys even when the sheet's label wording differs "
    "(e.g. 'Reg No', 'Vehicle No', 'Plate' all mean vehicle_registration_number; "
    "'KM', 'Odo' mean current_odometer_reading).\n"
    "- Normalise values: strip label noise; odometer/number fields as digits only; "
    "keep registration numbers as written.\n"
    "- 'description' must be a list of distinct reported problems/complaints "
    "(split numbered or multi-line complaints).\n"
    "- If a field is genuinely absent, use \"\" (or [] for description) and give it "
    "low confidence.\n"
    "- 'confidence' is an object mapping every field to a number 0..1 for how sure "
    "you are of that field's value.\n"
    "- Output ONLY minified JSON of the form "
    '{"data": {...schema fields...}, "confidence": {...}} with no prose, no code fences.'
)

def _schema_hint():
    return {
        "data": {
            **{f: "" for f in SCHEMA_FIELDS},
            "description": [],
        },
        "confidence": {**{f: 0 for f in SCHEMA_FIELDS}, "description": 0},
    }

def _build_prompt(textract_kv, extracted_text):
    return (
        _INSTRUCTIONS
        + "\n\nSchema shape (values are placeholders):\n"
        + json.dumps(_schema_hint())
        + "\n\nTextract key/values:\n"
        + json.dumps(textract_kv or {}, ensure_ascii=False)
        + "\n\nRaw text lines:\n"
        + json.dumps(extracted_text or [], ensure_ascii=False)
    )

def _extract_json(text):
    """Pull the first JSON object out of a model response, tolerating fences/prose."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

def _call_bedrock(prompt):
    """Primary: reuse the AWS credentials already used for Textract."""
    import boto3  # already a dependency (Textract)

    model_id = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    region = os.getenv("AWS_REGION", "us-east-1")

    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    parts = payload.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

def _call_anthropic(prompt):
    """Fallback: direct Anthropic API if ANTHROPIC_API_KEY is set."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("no ANTHROPIC_API_KEY")

    model = os.getenv("CHATBOT_MODEL", "claude-sonnet-5")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.loads(r.read())
    parts = payload.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

def _coerce(result, fallback_data):
    """Ensure a complete, well-typed schema regardless of what the model returned."""
    data = dict(result.get("data") or {})
    conf = dict(result.get("confidence") or {})

    out_data = {}
    out_conf = {}

    for field in SCHEMA_FIELDS:
        val = data.get(field)
        if val is None or val == "":
            val = (fallback_data or {}).get(field, "") or ""
            out_conf[field] = float(conf.get(field, 0.4)) if val else 0.0
        else:
            out_data_val = str(val).strip()
            val = out_data_val
            out_conf[field] = float(conf.get(field, 0.85))
        out_data[field] = val

    desc = data.get("description")
    if not isinstance(desc, list):
        desc = (fallback_data or {}).get("description")
        if not isinstance(desc, list):
            desc = [d for d in (str(desc).split("\n") if desc else []) if d.strip()]
    out_data["description"] = [str(d).strip() for d in desc if str(d).strip()]
    out_conf["description"] = float(conf.get("description", 0.8 if out_data["description"] else 0.0))

    return {"data": out_data, "confidence": out_conf}

def _rule_confidence(fallback_data):
    """Confidence map for the pure rule-based path: filled => medium, blank => 0."""
    conf = {}
    for field in SCHEMA_FIELDS:
        conf[field] = 0.6 if (fallback_data or {}).get(field) else 0.0
    desc = (fallback_data or {}).get("description") or []
    conf["description"] = 0.6 if desc else 0.0
    return conf

def map_job_card(textract_kv, extracted_text, fallback_data):
    """
    Returns: { "data": {...schema...}, "confidence": {...}, "mapping_source": str }

    Never raises: on any provider failure it falls back to the rule-based data
    that was passed in, so the OCR endpoint stays reliable.
    """
    prompt = _build_prompt(textract_kv, extracted_text)

    providers = [("bedrock", _call_bedrock), ("anthropic", _call_anthropic)]

    for name, call in providers:
        try:
            raw = call(prompt)
            parsed = _extract_json(raw)
            if parsed and isinstance(parsed.get("data"), dict):
                merged = _coerce(parsed, fallback_data)
                merged["mapping_source"] = name
                return merged
        except Exception as e:  # noqa: BLE001 - any failure -> try next / fallback
            print(f"[llm_mapper] {name} unavailable: {e}")

    fb = fallback_data or {}
    data = {f: (fb.get(f, "") or "") for f in SCHEMA_FIELDS}
    desc = fb.get("description")
    if not isinstance(desc, list):
        desc = [d for d in (str(desc).split("\n") if desc else []) if d.strip()]
    data["description"] = [str(d).strip() for d in desc if str(d).strip()]
    return {
        "data": data,
        "confidence": _rule_confidence(fb),
        "mapping_source": "rules",
    }
