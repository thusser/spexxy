import logging
import argparse
import os

from spexxy.grid import Grid
from spexxy.data import LSF, Spectrum, SpectrumFits

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # create parser
    parser = subparsers.add_parser('convert', help='Convert a grid from one type to another.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Input grid')
    subparsers = parser.add_subparsers()

    # to files grid
    parser_files = subparsers.add_parser('files', help='Convert to files grid')
    parser_files.set_defaults(func=_convert_to_files)
    parser_files.add_argument('outdir', type=str, help='Output path')
    parser_files.add_argument('-c', '--csv', type=str, help='CSV filename', default='grid.csv')
    pattern = 'Z-{FeH:.2f}.Alpha={Alpha:.2f}/lte{Teff:05.0f}{logg:+.2f}{FeH:+.2f}.Alpha={Alpha:.2f}.fits'
    parser_files.add_argument('-f', '--filenames', type=str, help='Filename pattern', default=pattern)


def _convert_to_files(args):
    # load grid
    grid = Grid.load(args.input)

    # does output directory exist?
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # open CSV file
    with open(os.path.join(args.outdir, args.csv), 'w') as csv:
        # write header
        csv.write('Filename,' + ','.join(grid.axis_names()) + '\n')

        # loop all parameters
        all_params = grid.all()
        for i, params in enumerate(all_params, 1):
            log.info('(%d/%d) %s...', i, len(all_params), params)

            # get spectrum
            spec = SpectrumFits(spec=grid(params))

            # get filename
            p = dict(zip(grid.axis_names(), params))
            filename = args.filenames.format(**p)

            # directory
            spec_filename = os.path.join(args.outdir, filename)
            spec_path = os.path.dirname(spec_filename)
            if not os.path.exists(spec_path):
                os.makedirs(spec_path)

            # save spectrum
            spec.save(spec_filename)

            # add to csv
            csv.write(filename + ',' + ','.join([str(p) for p in params]) + '\n')
