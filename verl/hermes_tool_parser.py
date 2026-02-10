import json
import logging

def extract_tool_calls(response):
    try:
        raw_function_calls = [
            json.loads(match[0]) for match in extract_matches(response)
        ]
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e} in response: {response}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e} in response: {response}")
        return []

    # Continue processing raw_function_calls
    return raw_function_calls