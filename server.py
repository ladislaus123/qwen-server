"""Runnable entrypoint for the Qwen service."""

from qwen_service.config import get_settings
from qwen_service.main import app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
