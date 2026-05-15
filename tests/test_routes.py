import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from qwen_service.config import Settings
from qwen_service.main import create_app


class FakeEngine:
    model_id = "fake-model"
    ready = True
    device = "test-device"

    async def load(self):
        return None

    async def close(self):
        return None

    async def generate(self, image, prompt, max_new_tokens):
        return "72.5"


class NotReadyEngine(FakeEngine):
    ready = False
    device = None


def _png_base64():
    image = Image.new("RGB", (4, 3), color=(0, 255, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_health_reports_ready_engine():
    app = create_app(
        settings=Settings(model_id="fake-model"),
        engine=FakeEngine(),
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["device"] == "test-device"


def test_health_reports_starting_engine():
    app = create_app(
        settings=Settings(model_id="fake-model"),
        engine=NotReadyEngine(),
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "starting"
    assert response.json()["ready"] is False


def test_analyze_returns_janus_compatible_response():
    app = create_app(
        settings=Settings(model_id="fake-model"),
        engine=FakeEngine(),
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={
                "image": _png_base64(),
                "prompt": "read it",
                "max_new_tokens": 100,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"success": True, "result": "72.5", "error": None}


def test_analyze_rejects_bad_image():
    app = create_app(
        settings=Settings(model_id="fake-model"),
        engine=FakeEngine(),
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={"image": "bad", "prompt": "read it"},
        )

    assert response.status_code == 400
