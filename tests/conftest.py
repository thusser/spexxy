import pytest
import numpy as np

from spexxy.grid import GridAxis, ValuesGrid


@pytest.fixture()
def number_grid():
    # define grid
    grid = np.array([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 7, 8]
    ])

    # define axes
    ax1 = GridAxis(name='x', values=list(range(grid.shape[1])))
    ax2 = GridAxis(name='y', values=list(range(grid.shape[0])))

    # combine values
    values = {}
    for x in ax1.values:
        for y in ax2.values:
            values[(x, y)] = grid[y, x]

    # return new grid
    return ValuesGrid([ax1, ax2], values)
