import json
import re


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

def parse_first_float(text):
    # Regular expression pattern to match a floating-point number anywhere in the string
    # The pattern matches optional sign, digits with optional decimal point and fractional part
    pattern = r'[+-]?(\d+(\.\d*)?|\.\d+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, convert the matched text to a float and return it
    if match:
        return float(match.group(0))
    else:
        # If no match is found, return None or raise an error as per your requirement
        return None

def extract_yes_or_no(text):    
    pattern = r"^\s*(yes|no)[\s!.,?]*"
    
    # Perform the regex search
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Extract the matched "yes" or "no" and capitalize the first letter
        return match.group(1).capitalize()
    else:
        # If there's no match, raise an error
        raise ValueError(f"Expected 'yes' or 'no', but got: {text}")