import base64
from io import BytesIO

import pytest
from PIL import Image

from qwen_service.image_io import ImageDecodeError, decode_base64_image


def _png_base64(size=(4, 3)):
    image = Image.new("RGB", size, color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_decodes_raw_base64_image():
    decoded = decode_base64_image(
        _png_base64(),
        max_pixels=100,
        max_bytes=100_000,
    )

    assert decoded.mode == "RGB"
    assert decoded.size == (4, 3)


def test_decodes_data_url_image():
    payload = f"data:image/png;base64,{_png_base64()}"

    decoded = decode_base64_image(payload, max_pixels=100, max_bytes=100_000)

    assert decoded.size == (4, 3)


def test_rejects_invalid_base64():
    with pytest.raises(ImageDecodeError, match="valid base64"):
        decode_base64_image("not base64!", max_pixels=100, max_bytes=100_000)


def test_rejects_non_image_payload():
    payload = base64.b64encode(b"hello").decode("utf-8")

    with pytest.raises(ImageDecodeError, match="supported image"):
        decode_base64_image(payload, max_pixels=100, max_bytes=100_000)


def test_rejects_oversized_image():
    with pytest.raises(ImageDecodeError, match="exceeds"):
        decode_base64_image(_png_base64(size=(20, 20)), max_pixels=100, max_bytes=100_000)
