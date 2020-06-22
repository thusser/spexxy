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
    parser = subparsers.add_parser('get', help='Get a spectrum from a grid or interpolator',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', type=float, help='Parameters for spectrum to get from grid', nargs='+')
    parser.add_argument('-o', '--output', type=str, help='Output fits file', default='spectrum.fits')
    parser.add_argument('-c', '--config', type=str, help='Config for grid', default='spexxy.yaml')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: get_spectrum_from_grid(**vars(args)))


def get_spectrum_from_grid(params: List[float], output: str, config: str, **kwargs):
    # create grid
    log.info('Initializing grid/interpolator...')
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    grid = create_object(config=cfg, log=log)
    if not isinstance(grid, Grid) and not isinstance(grid, Interpolator):
        raise ValueError('Object from config is neither a Grid nor an Interpolator.')

    # get spectrum
    log.info('Fetching spectrum at %s...', params)
    spec: Spectrum = grid(tuple(params))

    # write it
    log.info('Saving as %s...', output)
    SpectrumFits(spec=spec).save(output)
    log.info('Done.')
