import pytest

from spexxy.grid import GridAxis


class TestAxis(object):
    def test_neighbour(self):
        """ Test neighbours"""

        # create axis
        axis = GridAxis('test', [0, 2, 4, 6, 8, 10, 12])

        # on axis
        with pytest.raises(KeyError):
            axis.neighbour(6, -4)
        assert 0 == axis.neighbour(6, -3)
        assert 2 == axis.neighbour(6, -2)
        assert 4 == axis.neighbour(6, -1)
        assert 6 == axis.neighbour(6, 0)
        assert 8 == axis.neighbour(6, 1)
        assert 10 == axis.neighbour(6, 2)
        assert 12 == axis.neighbour(6, 3)
        with pytest.raises(KeyError):
            axis.neighbour(6, 4)

        # off axis
        with pytest.raises(KeyError):
            axis.neighbour(7.2, -4)
        assert 0 == axis.neighbour(7.2, -3)
        assert 2 == axis.neighbour(7.2, -2)
        assert 4 == axis.neighbour(7.2, -1)
        assert 6 == axis.neighbour(7.2, 0)
        assert 8 == axis.neighbour(7.2, 1)
        assert 10 == axis.neighbour(7.2, 2)
        assert 12 == axis.neighbour(7.2, 3)
        with pytest.raises(KeyError):
            axis.neighbour(7.2, 4)
