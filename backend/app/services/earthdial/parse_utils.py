# backend/app/services/earthdial/parse_utils.py
import re
from typing import List, Dict, Any

# Box pattern: <box>cx,cy,w,h,angle,label</box>
BOX_PATTERN = re.compile(
    r"<box>\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*(?:,\s*([^<>]+?)\s*)?</box>",
    re.IGNORECASE
)

def extract_caption(text):
    print("extract_caption received:", type(text), text)

    # Handle dict input
    if isinstance(text, dict):
        # Convert dict → readable caption
        try:
            # Example: {'airplane': [{'type': 'a220', 'color': 'white'}, ...]}
            parts = []
            for key, items in text.items():
                if isinstance(items, list):
                    for obj in items:
                        desc = ", ".join(f"{k}: {v}" for k, v in obj.items())
                        parts.append(f"{key} ({desc})")
                else:
                    parts.append(f"{key}: {items}")
            return "; ".join(parts) + "."
        except Exception as e:
            return str(text)

    # Normal string input
    if isinstance(text, str):
        text = text.strip()
        if not text:
            return ""

        for sep in ("\n\n", "\n", ". "):
            if sep in text:
                parts = text.split(sep)
                candidate = parts[0].strip()
                if len(candidate) > 10:
                    return candidate if candidate.endswith('.') else candidate + '.'
        return text

    # Fallback
    return str(text)
def normalize_output(output):
    """
    Converts dict model outputs into a readable string.
    Leaves strings unchanged.
    """
    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        parts = []
        for key, items in output.items():
            if isinstance(items, list):
                for obj in items:
                    if isinstance(obj, dict):
                        desc = ", ".join(f"{k}: {v}" for k, v in obj.items())
                        parts.append(f"{key} ({desc})")
                    else:
                        parts.append(f"{key}: {obj}")
            else:
                parts.append(f"{key}: {items}")
        return "; ".join(parts)

    # Fallback for weird types
    return str(output)

def extract_boxes_from_text(text: str) -> List[Dict[str, Any]]:
    matches = BOX_PATTERN.findall(text)
    results = []
    for m in matches:
        cx, cy, w, h, angle, label = (m[0], m[1], m[2], m[3], m[4], (m[5].strip() if m[5] else None))
        results.append({
            "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
            "angle": float(angle), "label": label
        })
    return results

def extract_binary_answer(text: str, question: str = None) -> str:
    """
    Try to find 'yes'/'no' near question terms. If not found, fallback to scanning for yes/no tokens.
    """
    txt = text.lower()
    if question:
        qwords = ' '.join(re.findall(r'\w+', question.lower()))
        idx = txt.find(qwords)
        if idx != -1:
            window = txt[max(0, idx-150): idx+300]
            if "yes" in window: return "Yes"
            if "no" in window: return "No"
    # general check
    if re.search(r'\b(yes|true|present|yep|yup)\b', txt): return "Yes"
    if re.search(r'\b(no|none|absent|not present|nope)\b', txt): return "No"
    return "Unknown"

def extract_numeric_answer(text: str, question: str = None) -> int:
    """
    Find counts related to question. Avoid numbers that are model labels (e.g. 'Boeing 737')
    Heuristic: if a number appears next to tokens like 'Boeing', 'model', 'type', skip it.
    Prefer numbers that near words like 'there are', 'count', 'visible', 'show'
    """
    txt = text
    # find candidate numbers along with surrounding context
    candidates = []
    for m in re.finditer(r'(\b\d+\b)', txt):
        num = int(m.group(1))
        start = max(0, m.start() - 30)
        end = min(len(txt), m.end() + 30)
        context = txt[start:end].lower()
        # skip if context contains 'boeing' or 'model' or 'series' (likely airplane model numbers)
        if any(x in context for x in ("boeing", "model", "series", "type")):
            continue
        candidates.append((num, context))
    # prefer candidates whose context contains words like 'there are', 'visible', 'found', 'count'
    for num, ctx in candidates:
        if any(x in ctx for x in ("there are", "visible", "found", "count", "how many", "detected")):
            return num
    # fallback to first candidate if any
    if candidates:
        return candidates[0][0]
    return None

def extract_semantic_answer(text: str, question: str = None) -> str:
    # heuristically return the phrase after the question if the model answers in sequence
    if not question:
        return text.strip()
    qt = question.strip().lower()
    txt = text.lower()
    idx = txt.find(qt[: min(len(qt), 60)])
    if idx != -1:
        # return 200 characters after question
        start = idx + len(qt)
        return text[start:start+300].strip()
    return text.strip()

