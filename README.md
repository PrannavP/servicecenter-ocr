Python Flask API for extracting text from uploaded images, built for service-center job cards.

To run: `pip install -r requirements.txt`, then run `app2.py` for the advanced extraction pipeline.

## Snap-to-Job-Card: AI field mapping

`POST /upload_v3` runs AWS Textract (FORMS) on the uploaded sheet, then maps the
raw key/values onto the canonical job-card schema. Instead of only matching
hard-coded labels, an LLM understands label variations ("Reg No", "Vehicle No",
"Plate" → `vehicle_registration_number`), normalises values, splits complaints
into a `description` list, and returns a **confidence score per field** so the
mobile app can highlight anything worth reviewing.

Response shape:

```json
{
  "filename": "...",
  "textract_key_value": { "...": "..." },
  "extracted_text": ["line 1", "line 2"],
  "data": { "customer_name": "...", "vehicle_registration_number": "...", "description": ["..."] },
  "confidence": { "customer_name": 0.92, "vehicle_registration_number": 0.5 },
  "mapping_source": "bedrock"
}
```

### Provider order (graceful fallback)

`helper/llm_mapper.py` tries, in order, and never fails the request:

1. **AWS Bedrock** (Claude) — reuses the same AWS credentials as Textract, so no
   extra key is needed once Bedrock model access is enabled.
2. **Anthropic API** — if `ANTHROPIC_API_KEY` is set.
3. **Rule-based** — the original `FIELD_MAP` parser (`mapping_source: "rules"`).

### Configuration

```
AWS_ACCESS_KEY_ID=...        # already used by Textract
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0   # optional
ANTHROPIC_API_KEY=sk-ant-...                                    # optional fallback
```

The Flutter app (`Service-Center-App`) consumes `data` + `confidence` in its
review screen, so users confirm/correct AI-filled fields before the job card is
created via the Node backend.
