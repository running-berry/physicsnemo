# from https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/lexicon/base.py

from collections.abc import Callable


class LexiconType(type):
    """Lexicon."""

    def __getitem__(cls, val: str) -> tuple[str, Callable]:
        """Retrieve variable name."""
        return cls.get_item(val)  # type: ignore[attr-defined]
