from qwen_service.config import Settings, get_settings


def test_clamp_max_new_tokens_uses_default():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(None) == 42


def test_clamp_max_new_tokens_caps_request():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(500) == 100


def test_clamp_max_new_tokens_raises_low_values_to_one():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(0) == 1


def test_default_vllm_concurrency_settings_are_conservative():
    settings = Settings()

    assert settings.vllm_max_num_seqs == 8
    assert settings.vllm_max_concurrent_requests == 8
    assert settings.vllm_gpu_memory_utilization == 0.85
    assert settings.vllm_cpu_offload_gb == 0.0
    assert settings.vllm_attention_backend is None


def test_local_vision_env_names_are_preferred(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("QWEN_MODEL_ID", "legacy/model")
    monkeypatch.setenv("LOCAL_VISION_MODEL_ID", "local/model")
    monkeypatch.setenv("LOCAL_VISION_MODEL_FAMILY", "auto")

    settings = get_settings()

    assert settings.model_id == "local/model"
    assert settings.model_family == "auto"
    get_settings.cache_clear()


def test_vllm_settings_are_loaded(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("LOCAL_VISION_BACKEND", "vllm")
    monkeypatch.setenv("LOCAL_VISION_VLLM_MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("LOCAL_VISION_VLLM_MAX_NUM_SEQS", "2")
    monkeypatch.setenv("LOCAL_VISION_VLLM_MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("LOCAL_VISION_VLLM_TENSOR_PARALLEL_SIZE", "4")
    monkeypatch.setenv("LOCAL_VISION_VLLM_GPU_MEMORY_UTILIZATION", "0.7")
    monkeypatch.setenv("LOCAL_VISION_VLLM_CPU_OFFLOAD_GB", "4")
    monkeypatch.setenv("LOCAL_VISION_VLLM_DTYPE", "float16")
    monkeypatch.setenv("LOCAL_VISION_VLLM_QUANTIZATION", "bitsandbytes")
    monkeypatch.setenv("LOCAL_VISION_VLLM_ATTENTION_BACKEND", "XFORMERS")

    settings = get_settings()

    assert settings.backend == "vllm"
    assert settings.vllm_max_model_len == 8192
    assert settings.vllm_max_num_seqs == 2
    assert settings.vllm_max_concurrent_requests == 3
    assert settings.vllm_tensor_parallel_size == 4
    assert settings.vllm_gpu_memory_utilization == 0.7
    assert settings.vllm_cpu_offload_gb == 4.0
    assert settings.vllm_dtype == "float16"
    assert settings.vllm_quantization == "bitsandbytes"
    assert settings.vllm_attention_backend == "XFORMERS"
    get_settings.cache_clear()


def test_vllm_max_model_len_can_be_disabled(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("LOCAL_VISION_VLLM_MAX_MODEL_LEN", "none")

    settings = get_settings()

    assert settings.vllm_max_model_len is None
    get_settings.cache_clear()
