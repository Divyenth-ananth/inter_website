from uuid import uuid4
from typing import Optional
from app.config import MODEL_PATH
from app.services.earthdial.earthdial_inference import EarthDialInference, CAPTION, HIGH_RGB_05_TOKEN

# Instantiate once at module load
earthdial = EarthDialInference(MODEL_PATH)

async def run_earthdial(image_pil, prompt):
    """
    Single turn chat helper.
    """
    # Ensure prompt has correct tokens if not present
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"
    
    # We use the internal helper from the class to run generation
    # Note: Accessing _run_inference is fine for this internal service wrapper
    text = earthdial._run_inference(
        image_pil, 
        prompt, 
        max_new_tokens=256
    )
    
    return {
        "request_id": str(uuid4()), 
        "output_text": text
    }

async def run_single_query(image_pil, question: str, query_type: Optional[str] = None,gsd: Optional[float] = None):
    if not query_type or query_type=="auto":
        query_type=earthdial.detect_query_type(question)
    return earthdial.infer_single(image_pil, question, query_type,gsd)

async def run_all_queries(image_pil, queries,input_image):
    """
    Runs the multi-task inference.
    Since earthdial.infer_multi now returns a clean Dictionary,
    we simply wrap it in the API response structure.
    """
    request_id = str(uuid4())

    # 1. Run Inference (Returns Dict)
    # The parsing happens inside EarthDialInference now
    structured_results = earthdial.infer_multi(image_pil, queries,input_image)

    # 2. Format Response
    # We construct the final JSON structure expected by your API schema
    return {
        "request_id": request_id,
        "queries": structured_results,
        "meta": {
            "model": "EarthDial_4B",
            "mode": "structured_multi_query"
        }
    }
