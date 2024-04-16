def parse_to_int(value):
    try:
        # First, convert to float, then to int
        return int(float(value))
    except ValueError:
        # Handle the case where conversion fails
        print(f"Warning: Could not convert '{value}' to int.")
        return None

def process_output(response):
    response = response.strip()
    if response.lower().startswith("no"):
        return False, []
    else:
        assert response.lower().startswith("yes")

        res = response.replace("<EOD>", "").strip()
        # look for '[' & ']'
        start_index = res.find("[")
        end_index = res.rfind("]") if start_index != -1 else -1

        # if no array is found, try to find single json objects
        if start_index == -1 or end_index == -1:
            start_index = res.find("{")
            end_index = res.rfind("}")
            if start_index != -1 and end_index != -1:
                json_string = (
                    "[" + res[start_index : end_index + 1] + "]"
                )  # Wrap the object in a list
            else:
                json_string = "[]"  # No JSON object or array found
        else:
            # get the json array
            json_string = res[start_index : end_index + 1]

        # output = json.loads(result[1].split("\n")[1].strip("Output: "))
        outputs = json.loads(json_string)

        return True, outputs