import os
import numpy as np
import pytest
from astropy.io import fits

from ..testdata import data_filename
from spexxy.weight import WeightRanges
from spexxy.data import Spectrum


class TestFromPath(object):
    def test_default(self):
        # filename for spectrum
        filename = data_filename('spectra/ngc6397id000010554jd2456865p5826f000.fits')

        # load spectrum and sigma
        spec = Spectrum.load(filename)

        # create weight
        weight = WeightRanges(ranges=[(4841., 4881., 10.)], initial=1.)
        w = weight(spec, filename)

        # first pixel should be one
        assert w[0] == 1.

        # get pixel at 4861AA and test it
        i = spec.index_of_wave(4861)
        assert w[i] == 10.
