import pytest
import numpy as np

from spexxy.data import Spectrum
from spexxy.component import StarComponent


class TestStar(object):
    def test_restframe(self):
        """Testing component vs spectrum."""

        # create spectrum
        wave = np.linspace(4500, 9300, 3500)
        spec = Spectrum(flux=np.random.rand(3500), wave=wave)

        # create component
        cmp = StarComponent(spec)

        # create params and interpolate
        spec_cmp = cmp(v=0., sig=0.)

        # test for equal
        assert np.array_equal(spec.wave, spec_cmp.wave)
        assert np.array_equal(spec.flux, spec_cmp.flux)

    def test_vrad(self):
        """Testing component vs spectrum with some vrad shift."""

        # create spectrum
        wave = np.linspace(4500, 9300, 3500)
        spec = Spectrum(flux=np.random.rand(3500), wave=wave)

        # create component
        cmp = StarComponent(spec)

        # redshift original spectrum
        spec.redshift(100.)

        # create params and interpolate
        spec_cmp = cmp(v=100., sig=0.)

        # test for equal
        assert np.array_equal(spec.wave, spec_cmp.wave)
        assert np.array_equal(spec.flux, spec_cmp.flux)
