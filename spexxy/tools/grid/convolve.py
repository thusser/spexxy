import logging
import argparse
import os

from spexxy.grid import Grid
from spexxy.data import LSF, Spectrum, SpectrumFits

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # create parser
    parser = subparsers.add_parser('convolve', help='Convolves all spectra in a grid',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Input grid')
    parser.add_argument('output', type=str, help='Output directory, in which a file grid is written')
    parser.add_argument('-w', '--wave', type=float, nargs=3, help='Start, end and step value for the wavelengths grid',
                        required=True)
    parser.add_argument('-o', '--overwrite', action="store_true", help='Overwrite existing files.')
    parser.add_argument('-l', '--log', action="store_true",
                        help='Create spectra on log scale, now -w and -f are given in log units.')
    parser.add_argument('-c', '--logconv', action="store_true", help='Do convolution on logarithmic scale.')
    parser.add_argument('-a', '--air', action="store_true", help='Convert spectra to air wavelengths.')
    parser.add_argument('-n', '--normalize', action="store_true", help='Normalize spectra to a mean flux of 1.')

    # group for fwhm and lsf
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--fwhm', metavar='fwhm', type=float, help='FWHM of gaussian to convolve spectra with')
    group.add_argument('-v', '--lsf', metavar='file', type=str, help='Name of file containing LSF')

    # argparse wrapper for create_grid
    def run(args):
        convolve_grid(args.input, args.output, args.wave[0], args.wave[1], args.wave[2],
                      overwrite=args.overwrite, vac2air=args.air, log_scale=args.log, log_convolve=args.logconv,
                      normalize=args.normalize, fwhm=args.fwhm, lsf_file=args.lsf)
    parser.set_defaults(func=run)


def convolve_grid(ingrid: str, outdir: str, wave_start: float, wave_end: float, sampling: float,
                  overwrite: bool = True, vac2air: bool = True, log_scale: bool = False,
                  log_convolve: bool = False, normalize: bool = False, fwhm: float = None,
                  lsf_file: str = None):

    # load grid
    grid = Grid.load(ingrid)

    # lsf?
    lsf = None if lsf_file is None else LSF.load(lsf_file)
    lsf_rescaled = False

    # outdir and grid file
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, 'grid.csv'), 'w') as f:
        f.write('Filename,' + ','.join(grid.axis_names()) + '\n')

    # get all params
    all_params = grid.all()

    # loop params
    for i, params in enumerate(all_params):
        log.info('(%d/%d) %s' % (i, len(all_params), params))

        # get spectrum
        log.info("  Loading spectrum...")
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

        # exists?
        if os.path.exists(outfile) and not overwrite:
            log.info('  Exists, skipping...')

        # log? does nothing, if modes already match
        spec.mode(Spectrum.Mode.LOGLAMBDA if log_convolve else Spectrum.Mode.LAMBDA)

        # process spec
        if fwhm:
            # constant sampling
            if spec.wave_step == 0.:
                spec = spec.resample_const()
            # convolve
            log.info("  Convolving with FWHM=%f...", fwhm)
            spec.smooth(fwhm)
        elif lsf is not None:
            # constant sampling
            if spec.wave_step == 0.:
                spec = spec.resample_const()
            # first spec?
            if not lsf_rescaled:
                log.info("  Resampling LSF...")
                lsf.wave_mode(spec.wave_mode)
                lsf.resample(spec)
                lsf_rescaled = True
            # convolve
            log.info("  Convolving with LSF...")
            spec = lsf(spec)

        # vac2air
        if vac2air:
            log.info("  Converting from vacuum to air wavelengths...")
            spec.vac_to_air()

        # log? both commands do nothing, if modes already match
        spec.mode(Spectrum.Mode.LOGLAMBDA if log_scale else Spectrum.Mode.LAMBDA)

        # resample
        wave_count = int((wave_end - wave_start) / sampling + 1.)
        log.info("  Resampling to final grid, wave_start=%f, wave_step=%f, wave_count=%d...",
                 wave_start, sampling, wave_count)
        spec = spec.resample(wave_start=wave_start, wave_step=sampling, wave_count=wave_count)

        # normalize?
        if normalize:
            spec.norm_to_mean()

        # save
        log.info("  Saving to {0:s}...".format(outfile))
        SpectrumFits(spec=spec).save(outfile)

        # add to CSV
        with open(os.path.join(outdir, 'grid.csv'), 'a') as f:
            csv.write(filename + ',' + ','.join([str(p) for p in params]) + '\n')


