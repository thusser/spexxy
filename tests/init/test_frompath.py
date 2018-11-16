import os
from astropy.io import fits
from spexxy.init import InitFromPath
from spexxy.component import Component


class create_testfile(object):
    def __init__(self, filename, results):
        self.filename = filename
        self.results = results

    def __enter__(self):
        # create file
        primary = fits.PrimaryHDU()

        # add results
        for cmp, values in self.results.items():
            for key, value in values.items():
                name = 'HIERARCH ANALYSIS %s %s' % (cmp.upper(), key.upper())
                primary.header[name] = value

        # write it
        primary.writeto(self.filename, overwrite=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.filename)


class TestFromPath(object):
    def test_default(self, tmpdir):
        # write test file
        with create_testfile(str(tmpdir / 'testfile.fits'), results={'star': {'v': 42., 'Teff': 5800}}):
            # create init
            init = InitFromPath(str(tmpdir))

            # init mock component
            cmp = Component('star')
            cmp.set('v')
            cmp.set('Teff')

            # run it
            init(cmp, 'testfile.fits')

            # compare
            assert 42 == cmp['v']
            assert 5800 == cmp['Teff']
