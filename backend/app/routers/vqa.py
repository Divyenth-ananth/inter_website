from fastapi import APIRouter, UploadFile, File, Form, HTTPException,Depends
from app.services.inference import run_all_queries
from app.models.vqa_request import VQARequest
from PIL import Image
import io
import json
import re # Needed for the fallback logic
from pydantic import BaseModel
from io import BytesIO
from app.services.inference import run_single_query
from uuid import uuid4
import requests
from app.dependencies import verify_api_key
from typing import Optional

router= APIRouter()
class SingleVQAResponse(BaseModel):
    request_id: str
    _type: str
    answer: str
    raw_output: str
    meta: dict

@router.post("/vqa/single", response_model=SingleVQAResponse)
async def vqa_single(
    image: UploadFile = File(...),
    question: str = Form(...),
    query_type: str = Form("auto"),   # optional: "auto", "caption", "binary", ..
    gsd: Optional[float] = Form(None)
):
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = await run_single_query(pil_image, question, query_type,gsd)
    print(result)
    keys=set(result.keys())
    possible={"response","binary","numeric","grounding","semantic","caption","missing_gsd"}
    intersection=possible.intersection(keys)
    if not intersection:
        # Fallback if something weird happens
        detected_type = "response"
    else:
        detected_type = intersection.pop()
    if detected_type == "missing_gsd":
        return SingleVQAResponse(
            request_id=str(uuid4()),
            _type="missing_gsd",
            answer="GSD required for area calculation",
            raw_output="",
            meta={"model": "EarthDial_4B", "mode": "single_query", "status": "input_required"},
        )

    return SingleVQAResponse(
        request_id=str(uuid4()),
        _type=detected_type,
        answer=str(result.get(detected_type)),
        raw_output=str(result.get(detected_type)),
        meta={"model": "EarthDial_4B", "mode": "single_query"},
    )
@router.post("/vqa/query")
async def vqa_query(request: str = Form(...), image: UploadFile = File(None)):
    # 1. Parse Request
    try:
        request_dict = json.loads(request)
        input_image = request_dict.get("input_image", {})
        # Ensure we pass the raw dictionary structure to inference
        queries = request_dict.get("queries", {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    image_id = None 
    image_url = None
    # 2. Process Image (Extract dimensions for metadata)
    try:
        if image and image.filename:
            image_bytes = await image.read()
            image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = image_pil.size
            image_id = image.filename
            image_url = input_image.get("image_url", "")
        elif "image_url" in input_image and input_image["image_url"]:
            image_url = input_image["image_url"]

            # Download image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise error for bad status codes

            image_bytes = response.content
            image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = image_pil.size
            image_id = input_image.get("image_id", "downloaded_image.png")
        elif "metadata" in input_image:
            # If we can't get the actual image, we can't process it
            raise HTTPException(
                status_code=400, 
                detail="Image must be provided either as upload or valid image_url"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Include either 'image' file or 'image_url' in request"
            )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 3. Run Inference
    # This returns the dictionary structure we built in the previous step
    result = await run_all_queries(image_pil, queries,input_image)
    
    # Extract raw results from the service
    inference_data = result.get("queries", {})
    
    # --- Caption Handling ---
    caption_data = inference_data.get("caption_query", {})
    caption_text = caption_data.get("response")

    # --- Grounding Handling (With Fallback & Reformatting) ---
    grounding_data = inference_data.get("grounding_query", {})
    grounding_boxes = grounding_data.get("response", [])

    # FALLBACK: If grounding is empty, check if they are hidden in the caption
    if not grounding_boxes and caption_text:
        # Regex to find list-like structures [[x,y,w,h,a]]
        fallback_matches = re.findall(r"\[\s*(\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*)\s*\]", caption_text)
        for m in fallback_matches:
            try:
                parts = [float(p.strip()) for p in m.split(",") if p.strip()]
                if len(parts) >= 4:
                    grounding_boxes.append(parts)
            except:
                continue

    # FORMATTING: Transform simple list [[...]] -> list of dicts [{"object-id":...}]
    formatted_grounding_response = []
    if grounding_boxes:
        for idx, box in enumerate(grounding_boxes):
            formatted_grounding_response.append({
                "object-id": str(idx + 1),
                "obbox": box
            })

    # --- Construct Final JSON Response ---
    response = {
        "input_image": {
            "image_id": image_id,
            "image_url": image_url or "", # Replace with actual URL if stored
            "metadata": {
                "width": width,
                "height": height,
                "spatial_resolution_m": 0.5 # Default for EarthDial_4B
            }
        },
        "queries": {
            "caption_query": {
                "instruction": caption_data.get("instruction"),
                "response": caption_text
            },
            "grounding_query": {
                "instruction": grounding_data.get("instruction"),
                "response": formatted_grounding_response
            },
            "attribute_query": inference_data.get("attribute_query", {})
        }
    }

    return response
