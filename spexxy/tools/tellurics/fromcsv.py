import os
import logging
import pandas as pd

from spexxy.data import SpectrumFits, LOSVD


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
    return s[0], s[1]


def add_parser(subparsers):
    """
    Adds 'spexxy tellurics get' command

    :param subparsers:  Subparser to attach new one to.
    """

    # init parser
    parser = subparsers.add_parser('fromcsv', help='Interpolates a tellurics spectrum with the given parameters.')
    parser.set_defaults(func=run)

    # grid
    parser.add_argument('csv', help='CSV file to read abundances from', type=str)

    # aliases for grids, only used for CSV files
    parser.add_argument('--alias', help='<alias>=<file> pairs for grids', type=pair, nargs='+')

    # radial velocity parameter
    parser.add_argument('--vrad', help='Parameter name for radial velocity', type=str)

    # line broadening parameter
    parser.add_argument('--sig', help='Parameter name for line broadening sigma', type=str)

    # vac- -> air?
    parser.add_argument('--vac2air', help='Convert vacuum to air wavelengths', action='store_true')

    # output file
    parser.add_argument('-o', '--output', help='Output file', type=str, default='tellurics.fits')


def run(args):
    """
    Takes a list of tellurics grids and abundances and outputs a tellurics spectrum

    :param args:    argparse namespace with
                    .alias:     alias/grid pairs
                    .output:    Output file for mean tellurics
                                (default: tellurics.fits)
    """

    # read csv, expects three columns Parameter,Value,Error
    logging.info('Loading CSV...')
    csv = pd.read_csv(args.csv, index_col=False)

    # create tellurics spectrum
    spec, hdr = create_tellurics_from_csv(csv, args.alias, args.vrad, args.sig, args.vac2air)

    # write to file
    logging.info('Writing tellurics to %s...', args.output)
    if os.path.exists(args.output):
        os.remove(args.output)
    spec.save(args.output, headers=hdr)


def create_tellurics_from_csv(csv, alias, vrad_col, sig_col, vac2air):
    from spexxy.interpolator.telluricinterpolator import TelluricInterpolator

    # aliases
    alias = dict(alias) if alias else {}

    # iterate rows
    grids = []
    abundances = []
    vrad = 0.
    sigma = None
    for row in csv.iterrows():
        # get parameter and value
        param = row[1].loc['Parameter']
        value = row[1].loc['Value']

        # in aliases?
        if param in alias:
            logging.info('Using abundance %s=%.2f.', alias[param], value)
            grids.append(alias[param])
            abundances.append(value)

        # vrad
        if vrad_col is not None and param == vrad_col:
            vrad = value

        # sigma
        if sig_col is not None and param == sig_col:
            sigma = value

    # create tellurics interpolator
    logging.info('Creating interpolator...')
    ip = TelluricInterpolator(grids)

    # interpolate spectrum
    logging.info('Interpolating spectrum...')
    spec = ip.spectrum(abundances, spectrum_class=SpectrumFits)

    # vrad?
    if sigma is not None:
        logging.info('Applying shift of vrad=%.2f km/s with a broading sigma=%.2f km/s...', vrad, sigma)
        losvd = LOSVD([vrad, sigma, 0., 0., 0., 0.])
        spec.flux = losvd(spec)
    elif vrad != 0:
        logging.info('Applying shift of vrad=%.2f km/s...', vrad)
        spec.redshift(vrad)

    # vac2air?
    if vac2air:
        logging.info('Converting vacuum to air wavelengths...')
        spec.vac_to_air()

    # add more headers
    hdr = {}
    for i, g in enumerate(grids):
        hdr['HIERARCH TELLURICS ' + g] = abundances[i]
    if vrad is not None:
        hdr['HIERARCH TELLURICS VRAD'] = vrad

    # return spectrum and header
    return spec, hdr
