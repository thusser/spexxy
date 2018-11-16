import numpy as np
import pytest

from spexxy.mask import MaskEnds


class TestEnds(object):
    def test_default(self, spectrum):
        # create mask
        func = MaskEnds(2)

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([False, False, True, True, True, True, True, False, False], m)

    def test_zero(self, spectrum):
        # create mask
        func = MaskEnds(0)

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([True, True, True, True, True, True, True, True, True], m)

    def test_negative(self, spectrum):
        # create mask, should fail!
        with pytest.raises(ValueError):
            func = MaskEnds(-5)
