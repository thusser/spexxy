import numpy as np
from lmfit.models import GaussianModel

from spexxy.data import LOSVD


class TestLOSVD(object):
    def test_kernel(self):
        """Test the losvd kernel."""

        # create an LOSVD object with (v, sig), ignore hermites
        losvd = LOSVD([42., 3.14, 0., 0., 0., 0.])

        # get optimal X array
        x = losvd.x(1e-6)

        # get kernel
        kernel = losvd.kernel(x)

        # sum of kernel should be close to 1
        assert abs(np.sum(kernel) - 1.) < 1e-10

        # fit gaussian
        model = GaussianModel()
        params = model.guess(kernel, x=x)
        result = model.fit(kernel, x=x, params=params)

        # we expect a center of 42 and a sigma of 3.14
        assert abs(result.params['center'] - 42.) < 1e-10
        assert abs(result.params['sigma'] - 3.14) < 1e-10
