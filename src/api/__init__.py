from typing import Any

__all__ = ["create_app", "app"]


def __getattr__(name: str) -> Any:
    if name in ("create_app", "app"):
        from src.api.main import app, create_app

        if name == "app":
            return app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")