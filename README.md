# Local Vision Model Service

This service provides a small HTTP API for local ROI analysis backed by a
Hugging Face vision-language model. The default model remains
`Qwen/Qwen2.5-VL-7B-Instruct`, but the service can be pointed at other
downloaded Hugging Face vision models through environment variables.

The HTTP contract is intentionally simple and compatible with the telemetry
app's local processing client.

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

The first run downloads the configured model, so it can take a while and needs
enough disk space for the model cache.

## Run

```bash
python server.py
```

The default URL is `http://localhost:6000`.

To use this from the telemetry app, set:

```bash
LOCAL_PROCESSOR_URL=http://localhost:6000
```

`JANUS_SERVER_URL` still works as a legacy telemetry fallback.

## Model Selection

Default Qwen2.5-VL path:

```bash
LOCAL_VISION_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
LOCAL_VISION_MODEL_FAMILY=qwen2_5_vl
```

Generic Hugging Face chat-style vision model path:

```bash
LOCAL_VISION_MODEL_ID=llava-hf/llava-1.5-7b-hf
LOCAL_VISION_MODEL_FAMILY=auto
LOCAL_VISION_AUTO_MODEL_CLASS=auto_vision2seq
```

Many newer models work with:

```bash
LOCAL_VISION_MODEL_FAMILY=auto
LOCAL_VISION_AUTO_MODEL_CLASS=auto_image_text_to_text
```

Some repositories require:

```bash
LOCAL_VISION_TRUST_REMOTE_CODE=true
```

The generic engine expects a chat-style vision-language model with a compatible
`AutoProcessor`. Model families differ, so some Hugging Face models may still
need a small model-specific engine.

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

- `LOCAL_VISION_MODEL_ID`: defaults to `Qwen/Qwen2.5-VL-7B-Instruct`
- `LOCAL_VISION_MODEL_FAMILY`: `qwen2_5_vl` or `auto`
- `LOCAL_VISION_AUTO_MODEL_CLASS`: `auto_image_text_to_text` or `auto_vision2seq`
- `LOCAL_VISION_HOST`: defaults to `0.0.0.0`
- `LOCAL_VISION_PORT`: defaults to `6000`
- `LOCAL_VISION_DEVICE`: `auto`, `cuda`, `mps`, or `cpu`
- `LOCAL_VISION_LOAD_MODEL_ON_STARTUP`: load the model during FastAPI startup
- `LOCAL_VISION_TRUST_REMOTE_CODE`: allow model repos with custom code
- `LOCAL_VISION_DEFAULT_MAX_NEW_TOKENS`: default generation token count
- `LOCAL_VISION_MAX_NEW_TOKENS_LIMIT`: hard cap for request token counts
- `LOCAL_VISION_MAX_IMAGE_PIXELS`: maximum decoded image dimensions
- `LOCAL_VISION_MAX_IMAGE_BYTES`: maximum request image payload size
- `LOCAL_VISION_MIN_PIXELS` / `LOCAL_VISION_MAX_PIXELS`: Qwen processor image resizing bounds
- `LOCAL_VISION_USE_FLASH_ATTENTION`: enable FlashAttention 2 on CUDA when installed

Legacy `QWEN_*` environment variables are still accepted as fallbacks.

## Tests

Most tests use a fake engine and do not load the model.

```bash
pytest
```
