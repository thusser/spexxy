import os
import pytest
from spexxy.init import InitFromCsv
from spexxy.component import Component


@pytest.fixture()
def test_csv(tmpdir):
    # create file
    filename = str(tmpdir / 'results.csv')
    with open(filename, 'w') as f:
        f.write('Filename,STAR TEFF,STAR V,DUMMY\n')
        f.write('a.fits,4800,100,10\n')
        f.write('b.fits,,-10,30\n')

    # return filename
    yield filename

    # clean up
    os.remove(filename)


class TestFromCsv(object):
    def test_default(self, test_csv):
        """Test default behaviour, filling all values from CSV."""

        # init
        init = InitFromCsv(test_csv)

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, 'a.fits')

        # compare
        assert 100 == cmp['v']
        assert 4800 == cmp['Teff']

    def test_missing_file(self, test_csv):
        """Testing case of filename not existing in CSV."""

        # init
        init = InitFromCsv(test_csv)

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, 'c.fits')

        # compare
        assert None == cmp['v']
        assert None == cmp['Teff']

    def test_missing_col(self, test_csv):
        """Testing case of missing column in CSV."""

        # init
        init = InitFromCsv(test_csv)

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')
        cmp.set('logg')

        # run it
        init(cmp, 'a.fits')

        # compare
        assert 100 == cmp['v']
        assert 4800 == cmp['Teff']
        assert None == cmp['logg']

    def test_missing_value(self, test_csv):
        """Testing case of missing value in CSV."""

        # init
        init = InitFromCsv(test_csv)

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, 'b.fits')

        # compare
        assert -10 == cmp['v']
        assert None == cmp['Teff']

    def test_parameters(self, test_csv):
        """Test user defined list of parameters to set."""

        # init
        init = InitFromCsv(test_csv, parameters=['teff'])

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, 'a.fits')

        # compare
        assert None == cmp['v']
        assert 4800 == cmp['Teff']

    def test_dummy(self, test_csv):
        """Test user defined parameter list with non-existing columns."""

        # init
        init = InitFromCsv(test_csv, parameters=['dummy'])

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, 'a.fits')

        # compare
        assert None == cmp['v']
        assert None == cmp['Teff']
