# Local Vision Model Service

This service provides a small HTTP API for local ROI analysis backed by a
local vision-language model. The default model remains
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

## vLLM Backend

The default backend is Transformers. To run the same `/analyze` route through
vLLM, install vLLM and change the backend in `.env`:

```bash
pip install vllm

LOCAL_VISION_BACKEND=vllm
LOCAL_VISION_DEVICE=cuda
LOCAL_VISION_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
LOCAL_VISION_MODEL_FAMILY=qwen2_5_vl
LOCAL_VISION_VLLM_MAX_MODEL_LEN=4096
LOCAL_VISION_VLLM_MAX_NUM_SEQS=8
LOCAL_VISION_VLLM_MAX_CONCURRENT_REQUESTS=8
LOCAL_VISION_VLLM_TENSOR_PARALLEL_SIZE=1
LOCAL_VISION_VLLM_GPU_MEMORY_UTILIZATION=0.85
LOCAL_VISION_VLLM_DTYPE=auto
LOCAL_VISION_VLLM_QUANTIZATION=
```

Then restart the service with:

```bash
python server.py
```

The HTTP API stays the same. vLLM schedules concurrent `/analyze` requests
internally, so throughput improves when clients send multiple requests at the
same time. A client that sends one request and waits before sending the next one
will still be processed sequentially from the service's point of view.

If vLLM fails during startup with `Device string must not be empty`, CUDA is not
visible to vLLM or vLLM failed automatic platform detection. Confirm `nvidia-smi`
works and that `torch.cuda.is_available()` is `True`; keep
`LOCAL_VISION_DEVICE=cuda` to force the vLLM engine device.

For a 16 GB GPU, first ensure no old model process is still holding VRAM. If the
model still does not fit, start with:

```bash
LOCAL_VISION_VLLM_MAX_NUM_SEQS=1
LOCAL_VISION_VLLM_MAX_CONCURRENT_REQUESTS=1
LOCAL_VISION_VLLM_MAX_MODEL_LEN=2048
LOCAL_VISION_VLLM_GPU_MEMORY_UTILIZATION=0.70
LOCAL_VISION_MAX_PIXELS=401408
```

Then raise concurrency only after the model loads successfully.

vLLM is loaded lazily by the vLLM backend, so the regular Transformers install
and test suite do not require it.

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
- `LOCAL_VISION_BACKEND`: `transformers` or `vllm`
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
- `LOCAL_VISION_VLLM_MAX_MODEL_LEN`: optional vLLM context length cap; use `none` to omit
- `LOCAL_VISION_VLLM_MAX_NUM_SEQS`: vLLM maximum concurrent sequences
- `LOCAL_VISION_VLLM_MAX_CONCURRENT_REQUESTS`: maximum `/analyze` requests admitted to vLLM at once
- `LOCAL_VISION_VLLM_TENSOR_PARALLEL_SIZE`: vLLM tensor parallel size
- `LOCAL_VISION_VLLM_GPU_MEMORY_UTILIZATION`: fraction of GPU memory vLLM may reserve
- `LOCAL_VISION_VLLM_DTYPE`: vLLM model dtype, for example `auto`, `float16`, or `bfloat16`
- `LOCAL_VISION_VLLM_QUANTIZATION`: optional vLLM quantization mode, or blank/`none`

Legacy `QWEN_*` environment variables are still accepted as fallbacks.

## Tests

Most tests use a fake engine and do not load the model.

```bash
pytest
```
