import pytest
from spexxy.grid import GridAxis, Grid


class TestGrid(object):
    def test_neighbour(self):
        """ Test neighbours"""
        x = GridAxis('x', [0, 2, 4, 6, 8, 10, 12])
        y = GridAxis('y', [1, 3, 5, 7, 9, 11, 13])
        grid = Grid(axes=[x, y])

        # on axis
        with pytest.raises(KeyError):
            grid.neighbour((4, 5), 1, -3)
        assert (4, 1) == grid.neighbour((4, 5), 1, -2)
        assert (4, 3) == grid.neighbour((4, 5), 1, -1)
        assert (4, 5) == grid.neighbour((4, 5), 1, 0)
        assert (4, 7) == grid.neighbour((4, 5), 1, 1)

        # off axis
        with pytest.raises(KeyError):
            grid.neighbour((4, 5.3), 1, -3)
        assert (4, 1) == grid.neighbour((4, 5.3), 1, -2)
        assert (4, 3) == grid.neighbour((4, 5.3), 1, -1)
        assert (4, 5) == grid.neighbour((4, 5.3), 1, 0)
        assert (4, 7) == grid.neighbour((4, 5.3), 1, 1)

        # both off axis, which shouldn't make a difference
        assert (6.7, 1) == grid.neighbour((6.7, 5.3), 1, -2)
