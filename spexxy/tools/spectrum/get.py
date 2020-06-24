import logging
import argparse
import sys
from typing import List

import yaml

from spexxy.component import SpectrumComponent
from spexxy.data import Spectrum, SpectrumFits
from spexxy.grid import Grid
from spexxy.interpolator import Interpolator
from spexxy.object import create_object


log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('get', help='Get a spectrum from a grid or interpolator',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', type=float, help='Parameters for spectrum to get from grid/interpolator/component',
                        nargs='+')
    parser.add_argument('-o', '--output', type=str, help='Output fits file', default='spectrum.fits')
    parser.add_argument('-c', '--config', type=str, help='Config for grid', default='spexxy.yaml')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: get_spectrum_from_grid(**vars(args)))


def get_spectrum_from_grid(params: List[float], output: str, config: str, **kwargs):
    # create grid
    log.info('Initializing grid/interpolator/component...')
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    obj = create_object(config=cfg, log=log)

    # what type is it?
    if isinstance(obj, Grid) or isinstance(obj, Interpolator):
        # log
        log.info('Created an instance of %s, which is %s.',
                 type(obj).__name__, 'a Grid' if isinstance(obj, Grid) else 'an Interpolator')

        # get spectrum
        log.info('Fetching spectrum at %s...', params)
        spec: Spectrum = obj(tuple(params))

    elif isinstance(obj, SpectrumComponent):
        # log
        log.info('Created an instance of %s, which is a SpectrumComponent.', type(obj).__name__)

        # set params
        for param, value in zip(obj.param_names, params):
            obj.set(param, value=value)

        # get spectrum
        spec = obj()

    else:
        log.error('Created object is not of any supported type.')
        sys.exit(1)

    # write it
    log.info('Saving as %s...', output)
    SpectrumFits(spec=spec).save(output)
    log.info('Done.')
