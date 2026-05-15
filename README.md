# Qwen2.5-VL Local Service

This service provides a Janus-compatible HTTP API backed by
`Qwen/Qwen2.5-VL-7B-Instruct`. It is designed to run beside the telemetry app
without modifying the telemetry repository.

## API

`POST /analyze`

```json
{
  "image": "base64_image_string",
  "prompt": "Extract the numeric value from this image",
  "max_new_tokens": 100
}
```

Success:

```json
{
  "success": true,
  "result": "42.5"
}
```

Runtime failure:

```json
{
  "success": false,
  "error": "Human-readable error"
}
```

Malformed images return HTTP `400`.

## Setup

```bash
cd ../qwen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The first run downloads `Qwen/Qwen2.5-VL-7B-Instruct`, so it can take a while
and needs enough disk space for the model cache.

## Run

```bash
python server.py
```

The default URL is `http://localhost:6000`.

To use this from the telemetry app, set:

```bash
JANUS_SERVER_URL=http://localhost:6000
```

## Health Check

```bash
curl http://localhost:6000/health
```

## Analyze Example

```bash
python - <<'PY'
import base64
import requests

with open("roi.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": image,
    "prompt": "Extract the numeric value shown in this image. Return only the number or null.",
    "max_new_tokens": 100,
}

print(requests.post("http://localhost:6000/analyze", json=payload, timeout=60).json())
PY
```

## Configuration

All configuration comes from environment variables or `.env`.

- `QWEN_MODEL_ID`: defaults to `Qwen/Qwen2.5-VL-7B-Instruct`
- `QWEN_HOST`: defaults to `0.0.0.0`
- `QWEN_PORT`: defaults to `6000`
- `QWEN_DEVICE`: `auto`, `cuda`, `mps`, or `cpu`
- `QWEN_LOAD_MODEL_ON_STARTUP`: load the model during FastAPI startup
- `QWEN_DEFAULT_MAX_NEW_TOKENS`: default generation token count
- `QWEN_MAX_NEW_TOKENS_LIMIT`: hard cap for request token counts
- `QWEN_MAX_IMAGE_PIXELS`: maximum decoded image dimensions
- `QWEN_MAX_IMAGE_BYTES`: maximum request image payload size
- `QWEN_MIN_PIXELS` / `QWEN_MAX_PIXELS`: Qwen processor image resizing bounds
- `QWEN_USE_FLASH_ATTENTION`: enable FlashAttention 2 on CUDA when installed

## Tests

Most tests use a fake engine and do not load the model.

```bash
pytest
```
