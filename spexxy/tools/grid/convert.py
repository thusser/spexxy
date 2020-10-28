import logging
import argparse
import os
import numpy as np
from astropy.io import fits

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

    # to fits grid
    parser_fits = subparsers.add_parser('fits', help='Convert to fits grid')
    parser_fits.set_defaults(func=_convert_to_fits)
    parser_fits.add_argument('output', type=str, help='Output file')


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


def _convert_to_fits(args):
    # load grid and get all parameters
    grid = Grid.load(args.input)
    all_params = grid.all()

    # load first spectrum and get its length
    spec = grid(all_params[0])
    nwave = len(spec)

    # Now we want to write a probably very large FITS file. There are instructions for that:
    # https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html
    # Unfortunately, it only describes, how to create an empty file, not how to fill it without loading it again,
    # which then loads the whole file into memory. So, in the end it still consumes all memory, but at least we create
    # the file beforehand.
    # TODO: Try to fix this.

    # create a dummy header
    data = np.zeros((100, 100), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header
    while len(header) < 10:
        header.append()  # Adds a blank card to the end

    # reset size, set some headers and write to file
    header['NAXIS1'] = nwave
    header['NAXIS2'] = len(all_params)
    header['CRVAL1'] = header['WSTART'] = (spec.wave_start, 'Wavelength at reference pixel')
    header['CDELT1'] = header['WSTEP'] = (spec.wave_step, 'Wavelength grid step size')
    header['CRPIX1'] = (spec.wave_step, 'Reference pixel coordinates')
    header['CTYPE1'] = ('AWAV' if spec.wave_mode == Spectrum.Mode.LAMBDA else 'AWAV-LOG', 'Type of wavelength grid')
    header.tofile(args.output, overwrite=True)

    # grow file to required size
    shape = tuple(header['NAXIS{0}'.format(ii)] for ii in range(1, header['NAXIS'] + 1))
    with open(args.output, 'rb+') as fobj:
        fobj.seek(len(header.tostring()) + (np.product(shape) * np.abs(header['BITPIX'] // 8)) - 1)
        fobj.write(b'\0')

    # now we can open the file for writing
    output = fits.open(args.output, mode='update', memmap=False)

    # loop all parameters
    all_params = grid.all()
    for i, params in enumerate(all_params, 1):
        log.info('(%d/%d) %s...', i, len(all_params), params)

        # get spectrum
        spec = SpectrumFits(spec=grid(params))

        # write to FITS file
        output[0].data[i - 1, :] = spec.flux

    # transpose parameter list, so that we have columns instead of rows
    params = list(map(list, zip(*all_params)))

    # create columns
    params_cols = [fits.Column(name=name, format='E', array=data) for name, data in zip(grid.axis_names(), params)]

    # add table with parameters
    hdu_table = fits.BinTableHDU.from_columns(params_cols)
    hdu_table.name = 'PARAMS'
    output.append(hdu_table)

    # flush file and close
    output.flush()
    output.close()
