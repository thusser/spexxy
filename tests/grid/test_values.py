import numpy as np


class TestValues(object):
    def test_axis_values(self, number_grid):
        assert np.array_equal(number_grid.axis_values(0), [0, 1, 2, 3, 4])
        assert np.array_equal(number_grid.axis_values(1), [0, 1, 2, 3])

    def test_call(self, number_grid):
        assert 1 == number_grid((0, 0))
        assert 5 == number_grid((3, 2))
        assert 6 == number_grid((2, 3))
        assert 7 == number_grid((4, 1))
