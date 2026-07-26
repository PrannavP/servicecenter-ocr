def get_kv_map(response):
    key_map = {}
    value_map = {}
    block_map = {}

    for block in response["Blocks"]:
        block_id = block["Id"]
        block_map[block_id] = block

        if block["BlockType"] == "KEY_VALUE_SET":
            if "KEY" in block["EntityTypes"]:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map

def get_text(block, block_map):
    text = ""

    if "Relationships" not in block:
        return text

    for relationship in block["Relationships"]:
        if relationship["Type"] != "CHILD":
            continue

        for child_id in relationship["Ids"]:
            word = block_map[child_id]

            if word["BlockType"] == "WORD":
                text += word["Text"] + " "

            elif word["BlockType"] == "SELECTION_ELEMENT":
                if word["SelectionStatus"] == "SELECTED":
                    text += "YES "

    return text.strip()

def find_value_block(key_block, value_map):
    if "Relationships" not in key_block:
        return None

    for relationship in key_block["Relationships"]:
        if relationship["Type"] != "VALUE":
            continue

        for value_id in relationship["Ids"]:
            return value_map[value_id]

    return None

def extract_key_value_pairs(response):

    key_map, value_map, block_map = get_kv_map(response)

    kv = {}

    for _, key_block in key_map.items():

        value_block = find_value_block(key_block, value_map)

        key = get_text(key_block, block_map)

        value = ""

        if value_block:
            value = get_text(value_block, block_map)

        kv[key] = value

    return kv