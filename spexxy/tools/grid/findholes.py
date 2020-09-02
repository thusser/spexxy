import logging
import numpy as np
import pandas as pd
import argparse

from spexxy.grid import Grid, FilesGrid

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('findholes', help='Find holes in a grid',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('grid', type=str, help='Name of grid CSV', default='grid.csv')
    parser.add_argument('output', type=str, help='Output file with holes', default='holes.csv')

    # argparse wrapper for fill_holes
    def run(args):
        grid = FilesGrid(args.grid)
        find_holes(grid, args.output)
    parser.set_defaults(func=run)


def find_holes(grid: Grid, output: str = None) -> pd.DataFrame:
    # loop dimensions
    holes = []
    for axis in range(grid.num_axes()):
        holes.extend(_find_holes_in_axis(grid, axis))

    # sort
    holes = list(sorted(set(holes)))

    # to dataframe
    df = pd.DataFrame(columns=grid.axis_names(), data=holes)

    # store it?
    if output:
        df.to_csv(output, index=False)

    # return it
    return df


def _find_holes_in_axis(grid: Grid, axis: int):
    # get values in this axis
    values = grid.axis_values(axis)

    # get values in all other axes
    other_values = [grid.axis_values(i) for i in range(grid.num_axes()) if i != axis]

    # meshgrid and loop
    holes = []
    mg = np.meshgrid(*other_values)
    for other_params in [tuple(t) for t in zip(*(x.flat for x in mg))]:
        # build list of params
        params = [tuple(list(other_params[:axis]) + [val] + list(other_params[axis:])) for val in values]

        # check for existence
        exists = [p in grid for p in params]

        # find first and last point in grid
        try:
            first = exists.index(True)
            last = len(exists) - exists[-1::-1].index(True) - 1
        except ValueError:
            # True is not in list
            continue

        # finally, find holes, which are all points that do not exists and lie between first and last
        holes.extend([p for i, p in enumerate(params) if first < i < last and not exists[i]])

    # finished
    return list(set(holes))
