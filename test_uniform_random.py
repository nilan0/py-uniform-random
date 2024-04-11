"""\
Compare the last 10 random ints or floats in a generated sequence of 2000 with the corresponding
values generated with the original C++ implementation.
"""

import pytest
import numpy as np
from uniform_random import UniformRandom


def test_random_float():
    expected = np.array([
        0.3604774,
        0.2054947,
        0.6783437,
        0.0682753,
        0.5367789,
        0.8238853,
        0.6762124,
        0.3165595,
        0.5438526,
        0.2469376
    ], dtype=np.float32)

    rnd = UniformRandom(161803398)
    generated = [rnd.next_float() for _ in range(2000)][-10::]

    assert np.allclose(generated, expected, rtol=0, atol=1e-7)


def test_random_int():
    expected = [
        774119346,
        441296546,
        1456731971,
        146620103,
        1152724115,
        1769280212,
        1452155131,
        679806423,
        1167914696,
        530294576
    ]

    rnd = UniformRandom(161803398)
    generated = [rnd.next_int(0, 0x7FFFFFFF) for _ in range(2000)][-10::]

    assert np.array_equal(generated, expected)
