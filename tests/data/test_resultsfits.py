import unittest
import os
from astropy.io import fits
import numpy as np

from spexxy.data.resultsfits import ResultsFITS


class TestResultsFITS(unittest.TestCase):
    """Unit tests for ResultsFITS class."""

    def setUp(self):
        # create a test fits file
        dat = np.arange(10)
        hdu = fits.PrimaryHDU(dat)
        if os.path.exists("test.fits"):
            os.remove("test.fits")
        hdu.writeto("test.fits")

    def tearDown(self):
        # delete test file
        if os.path.exists("test.fits"):
            os.remove("test.fits")

    def test_empty(self):
        """Test whether empty FITS file really contains no results."""

        # open file
        f = fits.open("test.fits")
        # create ResultsFITS object
        rf = ResultsFITS(f[0], "HIERARCH ANALYSIS TEST")
        # check keys
        self.assertEqual(len(rf.keys()), 0)
        # close
        f.close()

    def test_value(self):
        """Test to write a single value as result."""

        # open file
        f = fits.open("test.fits", mode="update")
        # create ResultsFITS object
        rf = ResultsFITS(f[0], "TEST")
        # set a value
        rf["pi"] = 3.14159
        # save
        f.flush()
        f.close()
        # load header
        hdr = fits.getheader("test.fits")
        # check
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST PI"], 3.14159)
        self.assertFalse("HIERARCH ANALYSIS TEST PI ERR" in hdr)

    @staticmethod
    def _create_test_file_err():
        """Create a test file with value/error entry"""

        # open file
        f = fits.open("test.fits", mode='update')
        # create ResultsFITS object
        rf = ResultsFITS(f[0], "TEST")
        # set a value
        rf["vrad"] = [132.12, 1.54]
        # save
        f.flush()
        f.close()

    def test_error(self):
        """Test value/error entry."""

        # create test file
        self._create_test_file_err()

        # load header
        hdr = fits.getheader("test.fits")

        # check
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST VRAD"], 132.12)
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST VRAD ERR"], 1.54)

    @staticmethod
    def _count_in_header(filename, token):
        """Load FITS file as text and cound occurences of 'token' in
        header.

        :param filename:    Name of FITS file.
        :param token:       Token to search for.
        """

        # read file
        with open(filename, 'r') as content_file:
            content = content_file.read()

        # extract header
        content = content[:content.find(' END ')]

        # count appearances of token
        return content.count(token)

    def test_update(self):
        """Test for changing a value/error pair."""

        # create test file with some data
        self._create_test_file_err()

        # open file
        f = fits.open("test.fits", mode="update")

        # create ResultsFITS object
        rf = ResultsFITS(f[0], "TEST")

        # update values
        rf["vrad"] = [42., 2.7]

        # save
        f.flush()
        f.close()

        # "HIERARCH ANALYSIS TEST VRAD" is supposed to
        # appear twice (value and err)
        cnt = self._count_in_header("test.fits",
                                    "HIERARCH ANALYSIS TEST VRAD")
        self.assertEqual(cnt, 2)

        # load header
        hdr = fits.getheader("test.fits")

        # check
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST VRAD"], 42.)
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST VRAD ERR"], 2.7)

    def test_delete(self):
        """Testing to delet an entry from the result set."""

        # create test file with some data
        self._create_test_file_err()

        # open file
        f = fits.open("test.fits", mode='update')

        # create ResultsFITS object and add second value
        rf = ResultsFITS(f[0], "TEST")
        rf["TEFF"] = 5789

        # save
        f.flush()
        f.close()

        # load again and delete vrad
        f = fits.open("test.fits", mode='update')
        rf = ResultsFITS(f[0], "TEST")
        del rf["VRAD"]
        f.flush()
        f.close()

        # check appearance count
        cnt = self._count_in_header("test.fits",
                                    "HIERARCH ANALYSIS TEST VRAD")
        self.assertEqual(cnt, 0)
        cnt = self._count_in_header("test.fits",
                                    "HIERARCH ANALYSIS TEST TEFF")
        self.assertEqual(cnt, 1)

        # load header
        hdr = fits.getheader("test.fits")

        # check
        self.assertEqual(hdr["HIERARCH ANALYSIS TEST TEFF"], 5789)
        self.assertRaises(KeyError, hdr.__getitem__, "VRAD")

        # and last but not least, open with ResultsFITS again
        f = fits.open("test.fits")
        rf = ResultsFITS(f[0], "TEST")
        self.assertEqual(rf["TEFF"], [5789, None])
        self.assertEqual(rf["VRAD"], None)
        f.close()

    def test_delete_err(self):
        """Testing to delete just an error and keeping the value."""

        # create test file with some data
        self._create_test_file_err()

        # open file
        f = fits.open("test.fits", mode='update')

        # create ResultsFITS object and delete error
        rf = ResultsFITS(f[0], "TEST")
        rf["VRAD"] = 3.14159
        f.flush()
        f.close()

        # check appearance count
        cnt = self._count_in_header("test.fits",
                                    "HIERARCH ANALYSIS TEST VRAD")
        self.assertEqual(cnt, 1)

        # and last but not least, open with ResultsFITS again
        f = fits.open("test.fits")
        rf = ResultsFITS(f[0], "TEST")
        self.assertEqual(rf["VRAD"], [3.14159, None])
        f.close()
