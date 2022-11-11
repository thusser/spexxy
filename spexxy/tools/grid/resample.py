import logging
import argparse
import os

from spexxy.grid import Grid
from spexxy.data import LSF, Spectrum, SpectrumFits

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # create parser
    parser = subparsers.add_parser('resample', help='Resample all spectra in a grid',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(func=run)
    parser.add_argument('input', type=str, help='Input grid')
    parser.add_argument('output', type=str, help='Output directory, in which a file grid is written')
    parser.add_argument('wave_start', type=float, help='Start of the wavelengths grid')
    parser.add_argument('wave_step', type=float, help='Step size for the wavelengths grid')
    parser.add_argument('wave_count', type=int, help='Length of the wavelengths grid')
    parser.add_argument('-m', '--mode', type=str, choices=['lambda', 'log'], default='lambda',
                        help='Target wavelength mode')


def run(args):
    resample_grid(args.input, args.output, args.wave_start, args.wave_step, args.wave_count,
                  Spectrum.Mode[args.mode.upper()])


def resample_grid(ingrid: str, outdir: str, wave_start: float, wave_step: float, wave_count: float,
                  mode: Spectrum.Mode):

    # load grid
    grid = Grid.load(ingrid)

    # outdir and grid file
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, 'grid.csv'), 'w') as csv:
        csv.write('Filename,' + ','.join(grid.axis_names()) + '\n')

    # get all params
    all_params = grid.all()

    # loop params
    for i, params in enumerate(all_params):
        log.info('(%d/%d) %s' % (i, len(all_params), params))

        # get spectrum
        spec = grid(params)

        # does spectrum have a filename or do we need to create one?
        if hasattr(spec, 'filename'):
            filename = spec.filename
            basepath = os.path.relpath(filename, os.path.dirname(ingrid))
        else:
            filename = 'spec_' + '_'.join(['%.2f' % p for p in params]) + '.fits'
            basepath = filename

        # get name and directory of output file
        outfile = os.path.join(outdir, basepath)
        outpath = os.path.dirname(outfile)

        # does dir exist?
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # change mode
        spec.mode(mode)

        # resample it
        resampled = spec.resample(wave_start=wave_start, wave_step=wave_step, wave_count=wave_count)

        # save
        log.info("  Saving to {0:s}...".format(outfile))
        SpectrumFits(spec=resampled).save(outfile)

        # add to CSV
        with open(os.path.join(outdir, 'grid.csv'), 'a') as csv:
            csv.write(filename + ',' + ','.join([str(p) for p in params]) + '\n')


