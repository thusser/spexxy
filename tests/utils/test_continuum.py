import numpy as np

from spexxy.utils import continuum


MAXRMS = 0.3


def data():
    # create polynomial of 3rd order
    x = np.linspace(0, 4, 1000)
    y = -(x - 2) ** 3 - (x - 2) ** 2 + 2 * (x - 2)

    # add noise
    yn = y + np.random.normal(0, 0.2, x.shape)

    # return it
    return x, y, yn


class TestContinuum(object):
    def test_continuum(self):
        # get test data
        x, y, yn = data()

        # create base continuum and fit
        cont = continuum.Continuum(poly='polynomial', poly_degree=10)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c)**2)) < MAXRMS

        # same with legendre
        cont = continuum.Continuum(poly='legendre', poly_degree=10)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c)**2)) < MAXRMS

        # and chebyshev
        cont = continuum.Continuum(poly='chebyshev', poly_degree=10)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c) ** 2)) < MAXRMS

    def test_valid(self):
        # get test data
        x, y, yn = data()

        # add some outliers
        valid = np.ones((len(x)), dtype=bool)
        valid[np.random.randint(0, len(x), 10)] = False
        yn[~valid] -= 1000

        # create base continuum and fit
        cont = continuum.Continuum(poly='polynomial', poly_degree=10)
        c = cont(x, yn, valid=valid)

        # RMS should be small
        assert np.sqrt(np.mean((y - c)**2)) < MAXRMS


class TestMaximumBin():
    def test_default(self):
        # get test data
        x, y, yn = data()

        # create base continuum and fit
        cont = continuum.MaximumBin(poly='spline', sbin=20, frac=0.5)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c) ** 2)) < MAXRMS

    def test_lines(self):
        # get test data
        x, y, yn = data()

        # add some outliers
        valid = np.ones((len(x)), dtype=bool)
        valid[np.random.randint(0, len(x), 10)] = False
        yn[~valid] -= 10

        # create base continuum and fit
        cont = continuum.MaximumBin(poly='spline', sbin=20, frac=0.5)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c) ** 2)) < MAXRMS


class TestSigmaClipping():
    def test_default(self):
        # get test data
        x, y, yn = data()

        # create SigmaClipping continuum and fit
        cont = continuum.SigmaClipping(poly='legendre', poly_degree=20)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c) ** 2)) < MAXRMS

    def test_lines(self):
        # get test data
        x, y, yn = data()

        # add some outliers
        valid = np.ones((len(x)), dtype=bool)
        valid[np.random.randint(0, len(x), 10)] = False
        yn[~valid] -= 10

        # create SigmaClipping continuum and fit
        cont = continuum.SigmaClipping(poly='legendre', poly_degree=20)
        c = cont(x, yn)

        # RMS should be small
        assert np.sqrt(np.mean((y - c) ** 2)) < MAXRMS


class TestRegions():
    def test_line(self):
        # create an absorption line
        x = np.linspace(6553, 6573, 1000)
        mu, sig = 6563, 1.
        y = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        # add noise
        yn = y + np.random.normal(0, 0.2, x.shape)

        # create SigmaClipping continuum and fit
        cont = continuum.Regions(poly='legendre', poly_degree=1, regions=[(6553, 6556), (6570, 6573)])
        c = cont(x, yn)

        # continuum should be close to zero, check only central pixels
        assert np.sqrt(np.mean(c ** 2)) < MAXRMS
