"""world2data_cam package."""

__all__ = ["main"]
__version__ = "0.1.0"


def main() -> None:
    """Package-level entrypoint."""
    from .cli import main as cli_main

    cli_main()
