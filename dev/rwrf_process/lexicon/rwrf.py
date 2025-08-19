from collections.abc import Callable

import numpy as np

from .base import LexiconType

"""
umet10: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  earth rotated u 

vmet10: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  earth rotated v 10m
    units:        m s-1

T2: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XY 
    stagger:      
    description:  TEMP at 2 M
    units:        K

PSFC: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XY 
    stagger:      
    description:  SFC PRESSURE
    units:        Pa

slp: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  Sea Level Pressure
    units:        hPa

pw: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  Precipitable Water
    units:        kg m-2

umet_p: <xarray.Variable (Time: 1, pres_bottom_top: 31, south_north: 450, west_east: 450)> Size: 25MB
[6277500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  earth rotated u
    units:        m s-1

vmet_p: <xarray.Variable (Time: 1, pres_bottom_top: 31, south_north: 450, west_east: 450)> Size: 25MB
[6277500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  earth rotated v
    units:        m s-1
    
z_p: <xarray.Variable (Time: 1, pres_bottom_top: 31, south_north: 450, west_east: 450)> Size: 25MB
[6277500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  geopotential height (MSL)
    units:        m
    
tk_p: <xarray.Variable (Time: 1, pres_bottom_top: 31, south_north: 450, west_east: 450)> Size: 25MB
[6277500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  Temperature in Kelvin
    units:        K

QVAPOR_p: <xarray.Variable (Time: 1, pres_bottom_top: 31, south_north: 450, west_east: 450)> Size: 25MB
[6277500 values with dtype=float32]
Attributes:
    FieldType:    104
    MemoryOrder:  XYZ
    stagger:      
    description:  Water vapor mixing ratio
    units:        kg kg-1

qpepre: <xarray.Variable (Time: 1, south_north: 450, west_east: 450)> Size: 810kB
[202500 values with dtype=float32]
Attributes:
    description:  Precipitation from QPEPRE, interpolated to RWRF grid
    units:        mm/hr
    
pres_levels: <xarray.Variable (pres_bottom_top: 31)> Size: 124B
[31 values with dtype=float32]
[1000.  975.  950.  925.  900.  875.  850.  825.  800.  775.  750.  700.
  650.  600.  550.  500.  450.  400.  350.  300.  250.  225.  200.  175.
  150.  125.  100.   70.   50.   30.   20.] (pressure levels in hPa)
"""


class RWRFLexicon(metaclass=LexiconType):
    """RWRF Lexicon"""

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create RWRF vocab dictionary"""
        variables = {
            "u10": "umet10",
            "v10": "vmet10",
            "t2m": "T2",
            "sp": "PSFC",
            "msl": "slp",
            "tcwv": "pw",
            "qpepre": "qpepre",
        }
        prs_levels = [
            50,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            850,
            925,
            1000,
        ]

        prs_names_to_id = {
            "u": "umet_p",
            "v": "vmet_p",
            "z": "z_p",
            "t": "tk_p",
            "q": "QVAPOR_p",
        }
        prs_variables = {}
        for id, variable in prs_names_to_id.items():
            for level in prs_levels:
                prs_variables[f"{id}{level}"] = f"{variable}"

        return {**variables, **prs_variables}

    VOCAB = build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Get item from RWRF vocabulary."""
        rwrf_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify data value (if necessary)."""
            return x

        return rwrf_key, mod
