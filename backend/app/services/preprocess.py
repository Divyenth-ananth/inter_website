import io
import numpy as np
from PIL import Image
import rasterio

MAX_SIZE = 2048    # limit size to avoid OOM on large TIFFs


def load_image_from_bytes(image_bytes: bytes, filename: str):
    """
    Loads an image from bytes.
    Supports JPEG, PNG, TIFF, GeoTIFF.
    """

    if filename.lower().endswith((".tif", ".tiff")):
        return load_geotiff(image_bytes)
    else:
        return load_regular_image(image_bytes)


def load_regular_image(image_bytes: bytes):
    """Load JPG/PNG into a Pillow image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = resize_if_needed(img)
    return np.array(img)   # HWC uint8


def load_geotiff(image_bytes: bytes):
    """Load a GeoTIFF using rasterio into an RGB numpy array."""
    with rasterio.MemoryFile(image_bytes) as memfile:
        with memfile.open() as dataset:
            # dataset.count = number of bands, e.g., 4, 8, 12

            # Choose bands: if not enough, repeat band 1
            if dataset.count >= 3:
                bands = [1, 2, 3]   # simple RGB selection
            else:
                bands = [1, 1, 1]   # fallback for grayscale TIFF

            arr = dataset.read(bands)     # shape (3, H, W)

            # Convert CHW → HWC for PIL/NumPy operability
            img = np.transpose(arr, (1, 2, 0)).astype(np.uint8)

            # Convert to PIL for easy resize
            pil_img = Image.fromarray(img)

            pil_img = resize_if_needed(pil_img)

            return np.array(pil_img)


def resize_if_needed(pil_img):
    """Resize very large images to max dimension."""
    w, h = pil_img.size
    max_dim = max(w, h)

    if max_dim > MAX_SIZE:
        scale = MAX_SIZE / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return pil_img


def prepare_for_model(np_img: np.ndarray):
    """
    Convert image numpy array (HWC uint8) to model-ready format.
    This depends on GeoLLaVA’s expected tensor format.
    """

    # Normalize 0-1
    img = np_img.astype("float32") / 255.0

    # Convert HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    return img  # ready for torch.from_numpy() later
