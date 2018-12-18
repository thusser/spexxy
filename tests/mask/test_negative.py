import numpy as np

from spexxy.mask import MaskNegative


class TestNegative(object):
    def test_default(self, spectrum):
        # create mask
        func = MaskNegative()

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal(spectrum.flux > 0, m)
