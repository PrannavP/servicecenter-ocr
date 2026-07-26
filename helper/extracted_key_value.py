
import re

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return " ".join(text.split())

FIELD_MAP = {
    "customer name": "customer_name",
    "customer address": "customer_address",
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

        if normalized.startswith("description"):
            desc = []

            if ":" in line:
                after = line.split(":", 1)[1].strip()
                after = re.sub(r'^\d+\.?\s*', '', after)
                if after:
                    desc.append(after)

            i += 1

            while i < len(lines):
                current = lines[i]
                current_normalized = normalize(current)

                if current_normalized in FIELD_MAP:
                    break

                if re.fullmatch(r"\d+\.?", current):
                    i += 1
                    continue

                current = re.sub(r'^\d+\.?\s*', '', current)

                if current:
                    desc.append(current)

                i += 1

            data["description"] = desc
            continue

        for field_name, json_key in FIELD_MAP.items():

            if normalized.startswith(field_name):

                value = ""

                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip(".")

                if not value and i + 1 < len(lines):
                    value = lines[i + 1].strip()

                    if normalize(value) in FIELD_MAP:
                        value = ""
                    else:
                        i += 1

                data[json_key] = value
                break

        i += 1

    return data