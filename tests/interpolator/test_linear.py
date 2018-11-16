import pytest

from spexxy.interpolator import LinearInterpolator


class TestLinear(object):
    def test_on_grid(self, number_grid):
        # create interpolator
        ip = LinearInterpolator(grid=number_grid)

        # test
        assert number_grid((2, 3)) == ip((2, 3))

    def test_in_grid(self, number_grid):
        # create interpolator
        ip = LinearInterpolator(grid=number_grid)

        # test first axis
        assert 6.2 == ip((2.2, 3))

        # test second axis
        assert 4.5 == ip((1, 2.75))

        # and both
        assert 6.5 == ip((3.5, 2.5))

    def test_out_grid(self, number_grid):
        # create interpolator
        ip = LinearInterpolator(grid=number_grid)

        # test
        with pytest.raises(KeyError):
            ip((9, 10))
        with pytest.raises(KeyError):
            ip((-1, 3))
