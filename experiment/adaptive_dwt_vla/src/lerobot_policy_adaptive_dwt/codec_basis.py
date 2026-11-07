"""Haar DWT basis matrices built from ``pywt`` (``mode='periodization'``).

``haar_matrix(L) @ x`` equals ``np.concatenate(pywt.wavedec(x, 'haar', mode='periodization', level=log2 L))``
so coefficient order is ``[cA_J, cD_J, cD_(J-1), ..., cD_1]``. The basis is
orthonormal, so the inverse is the transpose (equal to ``pywt.waverec``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pywt

WAVELET = "haar"
MODE = "periodization"


@lru_cache(maxsize=16)
def haar_matrix(length: int) -> np.ndarray:
    if length < 2 or length & (length - 1):
        raise ValueError(f"segment length must be a power of two >= 2, got {length}")
    level = length.bit_length() - 1
    cols = [np.concatenate(pywt.wavedec(e, WAVELET, mode=MODE, level=level)) for e in np.eye(length)]
    m = np.stack(cols, axis=1)
    m.setflags(write=False)
    return m


@lru_cache(maxsize=16)
def group_of(length: int) -> np.ndarray:
    """Wavelet group of each coefficient index: 0 = cA_J, 1 = cD_J, ..., J = cD_1."""
    g = np.array([0] + [int(np.floor(np.log2(c))) + 1 for c in range(1, length)])
    g.setflags(write=False)
    return g
