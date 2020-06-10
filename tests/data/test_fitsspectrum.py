import unittest
from astropy.io import fits
import os
import numpy as np

from spexxy.data.fitsspectrum import FitsSpectrum


class TestFitsSpectrum(unittest.TestCase):
    def setUp(self):
        # create a test fits file
        dat = np.arange(10)
        hdu = fits.PrimaryHDU(dat)
        hdu.header['EXTNAME'] = 'PRIMARY'
        hdu.header['CRVAL1'] = 3000.
        hdu.header['CDELT1'] = 1.
        if os.path.exists("test.fits"):
            os.remove("test.fits")
        hdu.writeto("test.fits")

    def tearDown(self):
        # delete test file
        if os.path.exists("test.fits"):
            os.remove("test.fits")

    def test_results(self):
        """Quick test for results()."""

        # open file, add results and close file
        fs = FitsSpectrum("test.fits", mode='rw')
        res = fs.results("TEST")
        res['pi'] = 3.14159
        fs.close()

        # check
        self.assertEqual(fits.getval("test.fits",
                                     "HIERARCH ANALYSIS TEST PI"),
                         3.14159)

    @staticmethod
    def create_test_spectrum():
        """Create a test FITS file with a fake spectrum."""
        # open file
        fs = FitsSpectrum("test.fits", mode='w')

        # create fake spectrum
        from spexxy.data.spectrum import SpectrumFitsHDU
        spec = SpectrumFitsHDU(flux=np.cos(np.arange(1000)), wave_start=3000, wave_step=1, primary=True)

        # set spectrum and close file
        fs['PRIMARY'] = spec
        fs.close()

    def test_spectrum(self):
        """Test whether we can load the test spectrum."""

        # open file
        with FitsSpectrum("test.fits") as fs:
            # check length
            self.assertEqual(len(fs), 1)

            # get spectrum, and check length
            spec = fs.spectrum
            self.assertEqual(len(spec), 10)

    def test_contains(self):
        """Test __contains__"""
        with FitsSpectrum("test.fits") as fs:
            self.assertTrue('PRIMARY' in fs)

    def test_create_spectrum(self):
        """Test setting a primary spectrum in a new file."""

        # open file
        fs = FitsSpectrum("test2.fits", mode='w')

        # create fake spectrum
        from spexxy.data.spectrum import SpectrumFitsHDU
        spec = SpectrumFitsHDU(flux=np.cos(np.arange(1000)), wave_start=3000, wave_step=1, primary=True)

        # set spectrum and close file
        fs[0] = spec
        fs.close()

        # open file with FitsSpectrum
        with FitsSpectrum("test2.fits", "r") as fs:
            # check length
            self.assertEqual(len(fs), 1)

            # get spectrum, and check length
            spec = fs.spectrum
            self.assertEqual(len(spec), 1000)

        # clean up
        os.remove("test2.fits")

    def test_overwrite_spectrum(self):
        """Test overwriting the primary spectrum in an existing file."""

        # open file
        fs = FitsSpectrum("test.fits", mode='rw')

        # create fake spectrum
        from spexxy.data.spectrum import SpectrumFitsHDU

        spec = SpectrumFitsHDU(flux=np.cos(np.arange(1000)), wave_start=1000, wave_step=0.5, primary=True)

        # set spectrum and close file
        fs[0] = spec
        fs.close()

        # open file with FitsSpectrum
        with FitsSpectrum("test.fits", "r") as fs:
            # check length
            self.assertEqual(len(fs), 1)

            # get spectrum, and check stuff
            spec = fs.spectrum
            self.assertEqual(len(spec), 1000)
            self.assertEqual(spec.wave_start, 1000)
            self.assertEqual(spec.wave_step, 0.5)

    def test_add_spectrum(self):
        """Test adding another spectrum (best fit, tellurics, etc)
        to existing file."""

        # create fake spectrum
        from spexxy.data.spectrum import SpectrumFitsHDU

        spec = SpectrumFitsHDU(flux=np.cos(np.arange(1000)), wave_start=1500, wave_step=0.25, primary=True)
        spec2 = SpectrumFitsHDU(flux=np.cos(np.arange(1000)), wave_start=1500, wave_step=0.25, primary=False)

        # open file
        with FitsSpectrum("test.fits", mode='w') as fs:
            # set spectrum and close file
            fs[0] = spec

        # append
        with FitsSpectrum("test.fits", mode='rw') as fs:
            # set spectrum and close file
            fs['TEST'] = spec2

        # pyfits check
        with fits.open("test.fits") as f:
            # need 2 HDUs
            self.assertEqual(len(f), 2)
            # get HDU names and check
            hdu_names = [hdu.header['EXTNAME'] for hdu in f if 'EXTNAME' in hdu.header]
            self.assertLessEqual(hdu_names, ['TEST'])

        # open file with FitsSpectrum
        with FitsSpectrum("test.fits", "r") as fs:
            # check length
            self.assertEqual(len(fs), 2)

            # get spectrum, and check stuff
            spec = fs['TEST']
            self.assertEqual(len(spec), 1000)
            self.assertEqual(spec.wave_start, 1500)
            self.assertEqual(spec.wave_step, 0.25)


if __name__ == '__main__':
    unittest.main()
