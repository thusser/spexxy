import logging
import argparse
import sys
from typing import List, Tuple, Dict
import yaml

from spexxy.component import SpectrumComponent
from spexxy.data import Spectrum, SpectrumFits
from spexxy.grid import Grid
from spexxy.interpolator import Interpolator
from spexxy.object import create_object

log = logging.getLogger(__name__)


def key_value_pair_or_value(opt: str):
    """Argument type for argparse, which accepts either a single float or a key=value pair.

    Args:
        opt: Command line parameter

    Returns:
        Single float or tuple containing key and value
    """

    # does it contain a =?
    pos = opt.find('=')
    if pos < 0:
        # single value?
        try:
            return float(opt)
        except ValueError:
            raise argparse.ArgumentTypeError('Value %s is not a float.' % opt)

    # split
    key = opt[:pos]
    value = opt[pos + 1:]

    # convert value to float
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value %s for parameter %s is not a float.' % (value, key))

    # and return
    return key, value


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('get', help='Get a spectrum from a grid or interpolator',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', type=key_value_pair_or_value, nargs='+',
                        help='Parameters for spectrum as list of values or key=value pairs')
    parser.add_argument('-o', '--output', type=str, help='Output fits file', default='spectrum.fits')
    parser.add_argument('-c', '--config', type=str, help='Config for grid', default='spexxy.yaml')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: get_spectrum_from_grid(**vars(args)))


def check_params(names: List[str], params: Dict[str, float]):
    """Checks whether all names show up in params and vice versa.

    Args:
        names: List of parameter names from grid/interpolator/component
        params: Dictionary with parameters from command line

    Raises:
        argparse.ArgumentTypeError if lists don't match
    """
    if sorted(names) != sorted(params.keys()):
        raise argparse.ArgumentTypeError('Given list of parameters does not match parameters '
                                         'in grid/interpolator/component.')


def get_spectrum_from_grid(params: List[Tuple[str, float]], output: str, config: str, **kwargs):
    """Fetches a spectrum with given parameters from the grid/interpolator/component given in
    config and stores it in output.

    Args:
        params: Parameters for spectrum
        output: Output filename
        config: Configuration for grid/interpolator/component
    """

    # list of values or list of tuples?
    if all([isinstance(p, float) for p in params]):
        # all floats
        pass
    elif all(isinstance(p, tuple) for p in params):
        # params to dict
        params = dict(params)
    else:
        raise argparse.ArgumentTypeError('Parameter list must either be a list of floats or of param=value pairs.')

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

        # need to make dict?
        if isinstance(params, list):
            # check length of list
            if len(params) != len(obj.axes()):
                raise argparse.ArgumentTypeError('Invalid number of parameters')

            # to dict
            params = dict(zip([ax.name for ax in obj.axes()], params))

        # check params
        check_params([ax.name for ax in obj.axes()], params)

        # get parameters
        p = [params[ax.name] for ax in obj.axes()]

        # get spectrum
        log.info('Fetching spectrum at %s...', ', '.join(['%s=%.2f' % x for x in params.items()]))
        spec: Spectrum = obj(tuple(p))

    elif isinstance(obj, SpectrumComponent):
        # log
        log.info('Created an instance of %s, which is a SpectrumComponent.', type(obj).__name__)

        # need to make dict?
        if isinstance(params, list):
            # check length of list
            if len(params) != len(obj.param_names):
                raise argparse.ArgumentTypeError('Invalid number of parameters')

            # to dict
            params = dict(zip(obj.param_names, params))

        # check params
        check_params(obj.param_names, params)

        # set params
        for param, value in params.items():
            obj.set(param, value=value)

        # get spectrum
        log.info('Fetching spectrum at %s...', ', '.join(['%s=%.2f' % x for x in params.items()]))
        spec = obj()

    else:
        log.error('Created object is not of any supported type.')
        sys.exit(1)

    # write it
    log.info('Saving as %s...', output)
    SpectrumFits(spec=spec).save(output)
    log.info('Done.')
