import logging
import argparse
from typing import List

import yaml

from spexxy.data import Spectrum, SpectrumFits
from spexxy.grid import Grid
from spexxy.interpolator import Interpolator
from spexxy.object import create_object


log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('info', help='Prints information about a given grid',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, help='Config for grid')
    parser.add_argument('-v', '--values', action='store_true', help='Show all values on axes')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: get_spectrum_from_grid(**vars(args)))


def get_spectrum_from_grid(config: str, values: bool, **kwargs):
    """Prints info about grid given in config.

    Args:
        config: Config to build grid from.
        values: If True, print all values per axis.
    """

    # create grid
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    grid = create_object(config=cfg, log=log)
    if not isinstance(grid, Grid) and not isinstance(grid, Interpolator):
        raise ValueError('Object from config is neither a Grid nor an Interpolator.')

    # get axes
    axes = grid.axes()
    print('Found %d axes: %s' % (len(axes), ' '.join([ax.name for ax in axes])))

    # loop axes
    for i, ax in enumerate(axes, 1):
        # print axis name
        print('%3d. %s' % (i, ax.name))

        # values?
        if ax.values is not None:
            print('     # steps: %d' % len(ax.values))
            if values:
                print('     values:  %s' % ','.join(['%.2f' % v for v in ax.values]))
        else:
            print('     continuous')

        # min/max?
        if ax.min is not None and ax.max is not None:
            print('     min/max: %.2f / %.2f' % (ax.min, ax.max))

        # initial
        if ax.initial is not None:
            print('     initial: %.2f' % ax.initial)

