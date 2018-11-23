import os
import numpy as np
import pytest
from astropy.io import fits

from ..testdata import data_filename
from spexxy.weight import WeightFromSigma
from spexxy.data import Spectrum


class TestFromPath(object):
    def test_default(self):
        # filename for spectrum
        filename = data_filename('spectra/ngc6397id000010554jd2456865p5826f000.fits')

        # load spectrum and sigma
        spec = Spectrum.load(filename)
        sigma = fits.getdata(filename, 'SIGMA')

        # create weight
        weight = WeightFromSigma()

        # compare
        diff = weight(spec, filename) - 1./sigma
        assert np.max(np.abs(diff)) < 1e-5

    def test_squared(self):
        # filename for spectrum
        filename = data_filename('spectra/ngc6397id000010554jd2456865p5826f000.fits')

        # load spectrum and sigma
        spec = Spectrum.load(filename)
        sigma = fits.getdata(filename, 'SIGMA')

        # create weight
        weight = WeightFromSigma(squared=True)

        # compare
        diff = weight(spec, filename) - 1./sigma**2
        assert np.max(np.abs(diff)) < 1e-5
