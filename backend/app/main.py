from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, base64
from app.routers.health import router as health_router
from app.routers.vqa import router as vqa_router

app = FastAPI(
    title="VQA Service",
    description="A Visual Question Answering (VQA) service that processes images and answers questions about them.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/v1")
app.include_router(vqa_router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port=8000,reload=True)
