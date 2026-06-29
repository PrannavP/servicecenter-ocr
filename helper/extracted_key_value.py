# This is an helper function which processes the extracted text by OCR and then maps it into key value pair data strucuture and returns.
# So i can map it in flutter app easily.

import re

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return " ".join(text.split())

# add the fields of the job card here so it will be easier to map and send kvp.
FIELD_MAP = {
    "customer name": "customer_name",
    "customer number": "customer_number",
    "vehicle type": "vehicle_type",
    "vehicle chasis number": "vehicle_chasis_number",
    "vehicle registration number": "vehicle_registration_number",
    "date time": "date_time",
    "current odometer reading": "current_odometer_reading",
    "fuel level": "fuel_level",
    "service type": "service_type",
    "expected delivery date time": "expected_delivery_date_time",
    "service advisor": "service_advisor",
    "helmet kept": "helmet_kept",
}

def parse_ocr_result(extracted_text):
    lines = [line.strip() for line in extracted_text if line.strip()]

    data = {value: "" for value in FIELD_MAP.values()}
    data["description"] = []

    i = 0

    while i < len(lines):
        line = lines[i]
        normalized = normalize(line)

        # Description (multi-line)
        if normalized.startswith("description"):
            desc = []

            # Same-line description
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                after = re.sub(r'^\d+\.?\s*', '', after)
                if after:
                    desc.append(after)

            i += 1

            while i < len(lines):
                current = lines[i]
                current_normalized = normalize(current)

                # Stop if another field begins
                if current_normalized in FIELD_MAP:
                    break

                # Ignore numbering (1, 2., 3...)
                if re.fullmatch(r"\d+\.?", current):
                    i += 1
                    continue

                current = re.sub(r'^\d+\.?\s*', '', current)

                if current:
                    desc.append(current)

                i += 1

            data["description"] = desc
            continue

        # Normal fields
        for field_name, json_key in FIELD_MAP.items():

            if normalized.startswith(field_name):

                value = ""

                # Same-line value
                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip(".")

                # Next-line value
                if not value and i + 1 < len(lines):
                    value = lines[i + 1].strip()

                    # Ensure next line isn't another field
                    if normalize(value) in FIELD_MAP:
                        value = ""
                    else:
                        i += 1

                data[json_key] = value
                break

        i += 1

    return data