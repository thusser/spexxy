import numpy as np

from spexxy.data import Spectrum


class TestSpectrum(object):
    def test_truediv(self, spectrum):
        """Tests __mul__ method.

        Checking, whether we can divide a spectrum by a number, an array, and another spectrum."""

        # divide by number
        result = spectrum / 10.
        assert np.array_equal(result, spectrum.flux / 10.)

        # divide by array
        div = np.arange(1., 10., 1.)
        result = spectrum / div
        assert np.array_equal(result, spectrum.flux / div)

        # divide by spectrum
        other = Spectrum(flux=div, wave=spectrum.wave, wave_mode=spectrum.wave_mode)
        result = spectrum / other
        assert np.array_equal(result, spectrum.flux / other.flux)

    def test_itruediv(self, spectrum):
        """Tests __itruediv__ method.

        We only do one test, since it's basically the same as __truediv__."""

        # divide by number
        flux = spectrum.flux.copy()
        spectrum /= 10.
        assert np.array_equal(flux / 10., spectrum.flux)

    def test_mul(self, spectrum):
        """Tests __mul__ method.

        Checking, whether we can multiply a spectrum with a number, an array, and another spectrum."""

        # multiply with number
        result = spectrum * 10.
        assert np.array_equal(result, spectrum.flux * 10.)

        # multiply with array
        div = np.arange(1., 10., 1.)
        result = spectrum * div
        assert np.array_equal(result, spectrum.flux * div)

        # multiply with spectrum
        other = Spectrum(flux=div, wave=spectrum.wave, wave_mode=spectrum.wave_mode)
        result = spectrum * other
        assert np.array_equal(result, spectrum.flux * other.flux)

    def test_imul(self, spectrum):
        """Tests __imul__ method.

        We only do one test, since it's basically the same as __mul__."""

        # multiply with number
        flux = spectrum.flux.copy()
        spectrum *= 10.
        assert np.array_equal(flux * 10., spectrum.flux)
        
    def test_add(self, spectrum):
        """Tests __add__ method.

        Checking, whether we can add a spectrum to a number, an array, and another spectrum."""

        # add number
        result = spectrum * 10.
        assert np.array_equal(result, spectrum.flux * 10.)

        # add array
        div = np.arange(1., 10., 1.)
        result = spectrum + div
        assert np.array_equal(result, spectrum.flux + div)

        # add spectrum
        other = Spectrum(flux=div, wave=spectrum.wave, wave_mode=spectrum.wave_mode)
        result = spectrum + other
        assert np.array_equal(result, spectrum.flux + other.flux)

    def test_iadd(self, spectrum):
        """Tests __iadd__ method.

        We only do one test, since it's basically the same as __add__."""

        # add number
        flux = spectrum.flux.copy()
        spectrum += 10.
        assert np.array_equal(flux + 10., spectrum.flux)
