import numpy as np

from spexxy.utils.clip_sigma import clip_sigma


class TestClipSigma(object):
    def test_default(self, tmpdir):
        # create random test array
        a = np.random.normal(100, 1., 1000)

        # put in some outliers
        o = [10, 542, 750]
        a[o] = 0.

        # clip it
        b = clip_sigma(a)

        # three outliers should be clipped
        assert len(a) - len(o) == len(b)
