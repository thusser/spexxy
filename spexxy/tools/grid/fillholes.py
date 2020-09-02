import logging
import os
from typing import Callable
import numpy as np
import pandas as pd
import argparse

from spexxy.data import SpectrumFits
from spexxy.interpolator import SplineInterpolator

from spexxy.grid import FilesGrid, Grid

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('fillholes', help='Fill holes in a file grid',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('grid', type=str, help='Name of grid CSV', default='grid.csv')
    parser.add_argument('holes', type=str, help='Name of holes CSV', default='holes.csv')
    parser.add_argument('output', type=str, help='Directory to store interpolated spectra in', default='interpolated/')

    # argparse wrapper for fill_holes
    def run(args):
        grid = FilesGrid(args.grid)
        holes = pd.read_csv(args.holes, index_col=False)
        fill_holes(grid, holes, args.output)
    parser.set_defaults(func=run)


def fill_holes(grid: Grid, holes: Grid, output: str):
    # create spline interpolator on grid
    ip = SplineInterpolator(grid, n=100)

    # loop holes
    for i, row in holes.iterrows():
        # build params
        params = tuple([row[key] for key in grid.axis_names()])
        log.info('(%d/%d) %s', i, len(holes), params)

        # interpolate
        try:
            spec = ip(params)
        except KeyError:
            log.warning('  Could not interpolate spectrum.')
            continue

        # create filename
        filename = os.path.join(output, '_'.join(['%.1f' % f for f in params]) + '.fits')

        # save it
        SpectrumFits(spec=spec).save(filename)
