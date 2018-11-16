import os
from astropy.io import fits

from spexxy.init import InitFromVhelio
from spexxy.component import Component


class create_testfile(object):
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers

    def __enter__(self):
        # create file
        primary = fits.PrimaryHDU()

        # add headers
        for key, value in self.headers.items():
            primary.header[key] = value

        # write it
        primary.writeto(self.filename, overwrite=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.filename)


class TestFromVhelio(object):
    def test_default(self, tmpdir):
        # write test file
        filename = str(tmpdir / 'testfile.fits')
        with create_testfile(filename, headers={'RA': 12, 'DEC': -30, 'DATE-OBS': '2018-10-17 12:00:00'}):
            # create init
            init = InitFromVhelio()

            # run it
            cmp = Component('star')
            init(cmp, filename)

            # compare
            assert abs(cmp['v'] + 11.83) < 0.1

    def test_file_missing(self, tmpdir):
        # filename should not exist
        f = tmpdir / 'testfile.fits'
        if f.exists():
            f.unlink()

        # create init
        init = InitFromVhelio()

        # run it
        cmp = Component('star')
        init(cmp, str(f))

        # v should not be set
        assert 'v' not in cmp.param_names

    def test_header_missing(self, tmpdir):
        # write test file
        filename = str(tmpdir / 'testfile.fits')
        with create_testfile(filename, headers={}):
            # create init
            init = InitFromVhelio()

            # run it
            cmp = Component('star')
            init(cmp, filename)

            # v should not be set
            assert 'v' not in cmp.param_names
