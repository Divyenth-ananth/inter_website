# EarthDial VQA Backend Service

A production-ready FastAPI backend for Visual Question Answering (VQA) on satellite imagery, powered by the **EarthDial** vision-language model. This service efficiently handles inference requests with GPU acceleration and is designed for scalable remote sensing workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
- [Development Guide](#development-guide)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

The EarthDial VQA backend provides a clean, RESTful HTTP API for sending satellite images and natural language questions to the EarthDial model, receiving structured answers with optional bounding box annotations. Key features:

- **High Performance** – Built on FastAPI with async I/O and GPU-accelerated inference
- **Modular Architecture** – Clean separation between routers, services, and model logic
- **Type-Safe** – Full Pydantic model validation for requests and responses
- **Remote-Sensing Ready** – Native support for GeoTIFF, PNG, and other satellite imagery formats
- **Well-Documented** – Interactive Swagger UI and comprehensive API documentation

---

## Environment Specifications

The backend has been developed and tested with the following environment:

- **Operating System:** Ubuntu 20.04 / 22.04
- **Python:** 3.10 or newer
- **GPU/CUDA:** CUDA 11.8+ (required for EarthDial model inference)

### Key Libraries

- **fastapi & uvicorn** – API framework and ASGI server
- **torch & transformers** – Model loading and inference for the EarthDial VLM
- **rasterio & pillow** – Satellite image loading and preprocessing (GeoTIFF, PNG)
- **pydantic** – Data validation and schema definitions

All dependencies are listed in `requirements.txt`.

---

## Project Structure

The repository is organized as follows:

```text
app/
  config.py              # Configuration (MODEL_PATH, API keys, etc.)
  main.py                # FastAPI app setup, CORS, router inclusion

  routers/
    vqa.py               # VQA endpoints: receives images + questions and returns answers
    health.py            # Health check endpoint to verify the API is up

  services/
    inference.py         # Service layer connecting routers to model inference
    preprocess.py        # Utilities for loading, normalizing, and resizing images

  earthdial/
    earthdial_inference.py  # EarthDial model loading and generation logic
    prompt_builder.py       # Prompt construction for the VLM
    parse_utils.py          # Regex utilities for bounding box extraction and text cleanup

  models/
    vqa_request.py       # Pydantic schema for incoming VQA requests
    vqa_response.py      # Pydantic schema for VQA responses

requirements.txt          # Python dependencies
payload.json              # Example request payload for testing
```

---

## Configuration

The backend expects the EarthDial model weights to be available on disk.

- **Environment variable:** `MODEL_PATH`
- **Default path (if unset):** `../models/EarthDial_4B_RGB`

You can either:

1. Export `MODEL_PATH` in your shell:

   ```bash
   export MODEL_PATH=/path/to/models/EarthDial_4B_RGB
   ```

2. Or update `app/config.py` to point to the correct weights directory.

Make sure the directory contains the EarthDial checkpoint and any required tokenizer/config files.

---

## Installation

1. **Clone the repository** (or copy the project folder to your machine).

2. (Optional but recommended) **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that **CUDA 11.8+** and compatible GPU drivers are installed so that PyTorch can use the GPU for inference.

---

## Running the Server

Once dependencies and model weights are set up, start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--host 0.0.0.0` exposes the service on all network interfaces.
- `--port 8000` sets the HTTP port.
- `--reload` enables auto-reload during development when code changes.

When the server is running, the interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs

You can use this UI to inspect endpoints, view request/response models, and send test requests.

---

## API Endpoints

### Health Check

- **Method:** `GET`
- **Path:** `/health`
- **Description:** Simple endpoint to confirm that the API is up and responding.

**Example response:**

```json
{
  "status": "ok"
}
```

### VQA Endpoint

- **Method:** `POST`
- **Path:** (see `app/routers/vqa.py`, typically something like `/vqa`)
- **Description:** Accepts an image (or reference to one) and a natural language question, then returns the model␙s answer and optional bounding boxes.

#### Request

The expected request schema is defined in `models/vqa_request.py`. A typical request includes:

- Image data (e.g., path, URL, or encoded content depending on your implementation)
- A text question
- Optional configuration parameters (e.g., max tokens, decoding options, bbox extraction flags)

A sample payload is provided in `payload.json`.

#### Response

The response schema is defined in `models/vqa_response.py`. A typical response includes:

- The generated **answer** text
- Optional **bounding boxes** and related metadata parsed from the model output

The end-to-end flow is:

1. `routers/vqa.py` receives the request and validates it using the Pydantic model.
2. `services/inference.py` handles preprocessing and calls into `earthdial/earthdial_inference.py`.
3. `earthdial/prompt_builder.py` constructs the text prompt for EarthDial.
4. The model generates an answer, which is cleaned and post-processed by `parse_utils.py`.
5. A structured `vqa_response` object is returned to the client.

---

## Development Notes

- **CORS & Middleware**: Configured in `app/main.py` to allow the frontend to communicate with this backend.
- **Image Preprocessing**: `services/preprocess.py` is the central place to modify image loading, normalization, resizing, or tiling logic for different satellite data sources.
- **Prompt Engineering**: To adjust how questions and image context are presented to EarthDial, edit `earthdial/prompt_builder.py`.
- **Output Parsing**: Bounding box formats, regex patterns, and text cleanup rules live in `earthdial/parse_utils.py`.

---

## Testing the API

You can quickly test the VQA endpoint using the provided `payload.json` file.

### Using cURL

```bash
curl -X POST "http://localhost:8000/vqa" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Adjust the contents of `payload.json` (image reference, question, options) to match your data.

### Using Postman or Similar Tools

1. Import or open `payload.json` as the request body.
2. Set the request method to `POST`.
3. Point the URL to `http://localhost:8000/vqa`.
4. Send the request and inspect the JSON response.

---

## Deployment Considerations

For production deployments, you may want to:

- Run Uvicorn behind a reverse proxy such as Nginx.
- Enable HTTPS termination and authentication/authorization.
- Add logging, metrics, and tracing around inference calls.
- Configure timeouts and request limits to protect the model server.

---

## License

Specify your project license here (e.g., MIT, Apache 2.0) if applicable.

