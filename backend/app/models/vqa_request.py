from pydantic import BaseModel
from typing import Optional, List, Dict


class ImageMetadata(BaseModel):
    width: Optional[int]
    height: Optional[int]
    spatial_resolution_m: Optional[float] = None


class InputImage(BaseModel):
    image_id: str
    image_url: Optional[str] = None
    metadata: Optional[ImageMetadata]


class CaptionQuery(BaseModel):
    instruction: str


class GroundingQuery(BaseModel):
    instruction: str


class AttributeBinary(BaseModel):
    instruction: str


class AttributeNumeric(BaseModel):
    instruction: str


class AttributeSemantic(BaseModel):
    instruction: str


class AttributeQuery(BaseModel):
    binary: AttributeBinary
    numeric: AttributeNumeric
    semantic: AttributeSemantic


class Queries(BaseModel):
    caption_query: CaptionQuery
    grounding_query: GroundingQuery
    attribute_query: AttributeQuery


class VQARequest(BaseModel):
    input_image: InputImage
    queries: Queries
