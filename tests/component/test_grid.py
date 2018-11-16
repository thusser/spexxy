import pytest
import numpy as np

from spexxy.data import Spectrum
from spexxy.grid import GridAxis, ValuesGrid
from spexxy.interpolator import SplineInterpolator
from spexxy.component import GridComponent


@pytest.fixture()
def spec_grid_1d():
    # create some random spectra
    wave = np.linspace(4500, 9300, 3500)
    specs = {
        (5500.,): Spectrum(flux=np.random.rand(3500), wave=wave),
        (5600.,): Spectrum(flux=np.random.rand(3500), wave=wave),
        (5700.,): Spectrum(flux=np.random.rand(3500), wave=wave),
        (5800.,): Spectrum(flux=np.random.rand(3500), wave=wave),
        (5900.,): Spectrum(flux=np.random.rand(3500), wave=wave),
        (6000.,): Spectrum(flux=np.random.rand(3500), wave=wave)
    }

    # define axis
    ax = GridAxis(name='Teff', values=[5500, 5600, 5700, 5800, 5900, 6000])

    # return new grid
    return ValuesGrid([ax], specs)


class TestGrid(object):
    def test_restframe(self, spec_grid_1d):
        """Testing component vs direct Spline interpolation.

        We get a random 1D spectrun grid, and create a spline interpolator on it. Interpolating from this
        should give the same result as getting a spectrum from a Grid component with the same parameters.
        """

        # create interpolator
        ip = SplineInterpolator(grid=spec_grid_1d)

        # interpolate
        res_ip = ip((5723.,))

        # create component
        cmp = GridComponent(ip)

        # create params and interpolate
        res_cmp = cmp(Teff=5723., v=0., sig=0.)

        # test for equal
        assert np.array_equal(res_ip.wave, res_cmp.wave)
        assert np.array_equal(res_ip.flux, res_cmp.flux)

    def test_vrad(self, spec_grid_1d):
        """Testing component vs direct Spline interpolation with a vrad shift.

        The same as the test before, but this time, both spectra should be shifted in vrad.
        """

        # create interpolator
        ip = SplineInterpolator(grid=spec_grid_1d)

        # interpolate
        res_ip = ip((5723.,))
        res_ip.redshift(100.)

        # create component
        cmp = GridComponent(ip)

        # create params and interpolate
        res_cmp = cmp(Teff=5723., v=100., sig=0.)

        # test for equal
        assert np.array_equal(res_ip.wave, res_cmp.wave)
        assert np.array_equal(res_ip.flux, res_cmp.flux)

    def test_vac2air(self, spec_grid_1d):
        """Testing component vs direct Spline interpolation with a conversion vac->air.

        The same as the first test again, but this time, both spectra should be air wavelength corrected.
        """

        # create interpolator
        ip = SplineInterpolator(grid=spec_grid_1d)

        # interpolate
        res_ip = ip((5723.,))
        res_ip.vac_to_air()

        # create component
        cmp = GridComponent(ip, vac_to_air=True)

        # create params and interpolate
        res_cmp = cmp(Teff=5723., v=0., sig=0.)

        # test for equal
        assert np.array_equal(res_ip.wave, res_cmp.wave)
        assert np.array_equal(res_ip.flux, res_cmp.flux)
