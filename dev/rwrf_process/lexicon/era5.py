from collections.abc import Callable

import numpy as np

from .base import LexiconType

LEVELS = [
    250,
    500,
    850,
    1000,
]


class ERA5Lexicon(metaclass=LexiconType):
    """ERA5 Lexicon

    Note
    ----
    Variable named based on ERA5 names, see CDS docs for resources.
    """

    VOCAB = {
        "u10m": "u10",
        "v10m": "v10",
        "t2m": "t2m",
        "tcwv": "tcwv",
        "mslp": "mslp",
        "sp": "sp",
    }
    VOCAB.update({f"u{level}": f"u{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"z{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"t{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"q{level}" for level in LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ERA5 vocabulary."""
        era5_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify name (if necessary)."""
            return x

        return era5_key, mod
