import os

from spexxy.data import SpectrumFits


def pair(arg):
    """Type for <key>=<value> pairs used in argparse

    :param arg: Value of parameter from argparse.
    :return:    List of two elements.
    """

    # split at '='
    s = arg.split('=')

    # check length
    if len(s) != 2:
        raise RuntimeError("Value must be given as <grid>=<abundance> pair.")

    # convert value to float
    return s[0], float(s[1])


def add_parser(subparsers):
    """
    Adds 'spexxy tellurics get' command

    :param subparsers:  Subparser to attach new one to.
    """

    # init parser
    parser = subparsers.add_parser('get', help='Interpolates a tellurics spectrum with the given parameters.')
    parser.set_defaults(func=run)

    # grid
    parser.add_argument('abundances', help='<file>=<value> pairs for molecular abundances', type=pair, nargs='+')

    # output file
    parser.add_argument('-o', '--output', help='Output file', type=str, default='tellurics.fits')


def run(args):
    """
    Takes a list of tellurics grids and abundances and outputs a tellurics spectrum

    :param args:    argparse namespace with
                    .models:    grid/abundance pairs
                    .output:    Output file for mean tellurics
                                (default: tellurics.fits)
    """
    from spexxy.interpolator.telluricinterpolator import TelluricInterpolator

    # get all grids and abundances
    grids, abundances = zip(*args.abundances)

    # create tellurics interpolator
    ip = TelluricInterpolator(grids)

    # interpolate spectrum
    spec = ip.spectrum(abundances, spectrum_class=SpectrumFits)

    # save spectrum
    if os.path.exists(args.output):
        os.remove(args.output)
    spec.save(args.output)
