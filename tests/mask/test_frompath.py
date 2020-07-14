import os

import numpy as np
from astropy.io import fits

from spexxy.mask import MaskFromPath


class create_maskfile(object):
    def __init__(self, filename, len=9):
        self.filename = filename
        self.len = len

    def __enter__(self):
        # mock data
        data = np.array([False, False, True, True, True, True, True, False, False])

        # create file
        primary = fits.PrimaryHDU()
        goodpixels = fits.ImageHDU(data[:self.len].astype(np.short))
        goodpixels.header['EXTNAME'] = 'GOODPIXELS'
        hdus = fits.HDUList([primary, goodpixels])
        hdus.writeto(self.filename, overwrite=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.filename)


class TestFromPath(object):
    def test_default(self, spectrum, tmpdir):
        # write test file
        with create_maskfile(str(tmpdir / 'maskfile.fits')):
            # create mask
            func = MaskFromPath(str(tmpdir))

            # apply it
            m = func(spectrum, filename='maskfile.fits')

            # compare
            assert np.array_equal([False, False, True, True, True, True, True, False, False], m)

    def test_wrong_length(self, spectrum, tmpdir):
        # write test file
        with create_maskfile(str(tmpdir / 'maskfile.fits'), len=4):
            # create mask
            func = MaskFromPath(str(tmpdir))

            # apply it
            m = func(spectrum, filename='maskfile.fits')

            # result should be an empty mask
            assert np.array_equal([True, True, True, True, True, True, True, True, True], m)

    def test_file_missing(self, spectrum, tmpdir):
        # filename should not exist
        f = tmpdir / 'maskfile.fits'
        if f.exists():
            f.unlink()

        # create mask
        func = MaskFromPath(str(tmpdir))

        # apply it
        m = func(spectrum, filename='maskfile.fits')

        # result should be an empty mask
        assert np.array_equal([True, True, True, True, True, True, True, True, True], m)