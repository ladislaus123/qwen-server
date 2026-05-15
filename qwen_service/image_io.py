"""Image decoding and validation helpers."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when a request image cannot be decoded safely."""


def _extract_base64_payload(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ImageDecodeError("image must not be empty")

    if stripped.startswith("data:"):
        header, separator, payload = stripped.partition(",")
        if not separator or ";base64" not in header:
            raise ImageDecodeError("data URL must contain a base64 payload")
        stripped = payload

    return "".join(stripped.split())


def decode_base64_image(
    value: str,
    *,
    max_pixels: int,
    max_bytes: int,
) -> Image.Image:
    """Decode a base64 image into an RGB Pillow image."""
    payload = _extract_base64_payload(value)

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("image is not valid base64") from exc

    if not raw:
        raise ImageDecodeError("image payload is empty")
    if len(raw) > max_bytes:
        raise ImageDecodeError(f"image payload exceeds {max_bytes} bytes")

    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ImageDecodeError("image dimensions are invalid")
            if width * height > max_pixels:
                raise ImageDecodeError(f"image exceeds {max_pixels} pixels")

            image.load()
            return image.convert("RGB")
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("image payload is not a supported image") from exc
