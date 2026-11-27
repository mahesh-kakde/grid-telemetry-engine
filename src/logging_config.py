import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging format for CLI runners and service entrypoints."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )