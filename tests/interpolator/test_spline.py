import pytest

from spexxy.interpolator import SplineInterpolator


class TestSpline(object):
    def test_on_grid(self, number_grid):
        # create interpolator
        ip = SplineInterpolator(grid=number_grid)

        # test
        assert number_grid((2, 3)) == ip((2, 3))
