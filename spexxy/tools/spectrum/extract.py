import glob
import logging
import argparse
import os
from typing import List

from spexxy.data import FitsSpectrum, SpectrumFitsHDU

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('extract', help='Extract wavelength range from spectrum',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('start', type=float, help='Start of wavelength range')
    parser.add_argument('end', type=float, help='End of wavelength range')
    parser.add_argument('input', type=str, help='Input spectrum', nargs='+')
    parser.add_argument('-p', '--prefix', type=str, help='Output file prefix', default='extracted_')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: extract_spectra(**vars(args)))


def extract_spectra(input: List[str], prefix: str, start: float, end: float, **kwargs):
    # get list of filenames
    files = []
    for f in input:
        if '*' in input or '?' in input:
            files.extend(glob.glob(f))
        else:
            files.append(f)

    # make unique
    files = sorted(list(set(files)))

    # loop files
    for i, f in enumerate(files, 1):
        print('(%d/%d) %s' % (i, len(files), f))
        extract_spectrum(f, prefix, start, end)


def extract_spectrum(input: str, prefix: str, start: float, end: float):
    """Extracts the given wavelength range from input and store it as output.

    Args:
        input: Input filename
        prefix: Output filename prefix
        start: Wavelength start
        end: Wavelength end
    """

    # load spectrum
    with FitsSpectrum(input) as fs_in:
        # open spectrum to write
        with FitsSpectrum(prefix + os.path.basename(input), 'w') as fs_out:
            # copy extracted spectrum
            fs_out.spectrum = fs_in.spectrum.extract(start, end)

            # loop all extensions:
            for ext in fs_in.hdu_names():
                # skip no name
                if ext.strip() == '':
                    continue

                # try to get hdu
                try:
                    # get hdu
                    hdu = fs_in[ext]

                    # need to create new hdu with primary=False, since that gets lost on extract
                    fs_out[ext] = SpectrumFitsHDU(spec=hdu.extract(start, end), primary=False)

                except ValueError:
                    pass