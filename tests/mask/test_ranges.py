import numpy as np
import pytest
import os
from astropy.io import fits

from spexxy.mask import MaskRanges


class MockComponent(object):
    def __init__(self, vrad):
        self.vrad = vrad

    def __getitem__(self, item):
        return self.vrad


@pytest.fixture()
def fits_file(tmpdir):
    # create filename
    filename = str(tmpdir / 'test.fits')

    # write dummy fits file
    hdu = fits.PrimaryHDU(np.arange(10))
    hdu.header['HIERARCH COMBINE MEANVRAD'] = 100000.
    hdu.writeto(filename, overwrite=True)

    # yield filename
    yield filename

    # clean up
    os.remove(filename)


class TestRanges(object):
    def test_no_shift(self, spectrum):
        # ranges to mask
        ranges = [(2.2, 3.1), (7.6, 8.1)]

        # create mask
        func = MaskRanges(ranges, component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(0.)}})

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([True, True, False, True, True, True, True, False, True], m)

    def test_shift(self, spectrum):
        # ranges to mask
        ranges = [(2.2, 3.1), (7.6, 8.1)]

        # create mask
        func = MaskRanges(ranges, component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(100000.)}})

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([True, True, False, False, True, True, True, True, True], m)

    def test_wrong_order(self, spectrum):
        # ranges to mask
        ranges = [(6.1, 3.2)]

        # create mask
        func = MaskRanges(ranges, component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(0.)}})

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([True, True, True, False, False, False, True, True, True], m)

    def test_bounds(self, spectrum):
        # ranges to mask
        ranges = [(-10, 3.2)]

        # create mask
        func = MaskRanges(ranges, component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(0.)}})

        # apply it
        m = func(spectrum)

        # compare
        assert np.array_equal([False, False, False, True, True, True, True, True, True], m)

    def test_extra_shift(self, spectrum, fits_file):
        """Test extra shifts. The given value in the mock FITS file is 100000."""

        # ranges to mask
        ranges = [(2.2, 3.1), (7.6, 8.1)]

        # create mask, apply it and test, should give same result as test_shift()
        func = MaskRanges(ranges, vrad='HIERARCH COMBINE MEANVRAD', component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(0.)}})
        m = func(spectrum, fits_file)
        assert np.array_equal([True, True, False, False, True, True, True, True, True], m)

        # create mask, apply it and test, should give same result as test_no_shift()
        func = MaskRanges(ranges, vrad='HIERARCH COMBINE MEANVRAD', component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(-100000.)}})
        m = func(spectrum, fits_file)
        assert np.array_equal([True, True, False, True, True, True, True, False, True], m)

        # and with a number
        func = MaskRanges(ranges, vrad=500000., component='star', vrad_parameter='v',
                          objects={'components': {'star': MockComponent(-500000.)}})
        m = func(spectrum, fits_file)
        assert np.array_equal([True, True, False, True, True, True, True, False, True], m)