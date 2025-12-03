from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class CaptionResponse(BaseModel):
    instruction: str
    response: str


class GroundingObject(BaseModel):
    object_id: str
    obbox: List[float]


class GroundingResponse(BaseModel):
    instruction: str
    response: List[GroundingObject]


class AttributeResponse(BaseModel):
    instruction: str
    response: Any


class VQAResponse(BaseModel):
    request_id: str
    queries: Dict[str, Any]
    meta: Dict[str, Any]
