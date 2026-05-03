import re

def extract_xml_answer(llm_response):
    """
    Extracts the numerical value from within <answer>...</answer> tags.
    Returns None if tags are missing or empty.
    """
    match = re.search(r"<answer>(.*?)</answer>", llm_response, re.IGNORECASE | re.DOTALL)
    if match:
        # Strip out any formatting like commas or spaces (e.g., "1,000" -> "1000")
        raw_val = match.group(1).strip()
        clean_val = re.sub(r'[^\d\.\-]', '', raw_val)
        return clean_val
    return None