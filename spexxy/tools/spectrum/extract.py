import logging
import argparse
import sys
from typing import List, Tuple, Dict
import yaml

from spexxy.component import SpectrumComponent
from spexxy.data import Spectrum, SpectrumFits, FitsSpectrum, SpectrumFitsHDU
from spexxy.grid import Grid
from spexxy.interpolator import Interpolator
from spexxy.object import create_object

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('extract', help='Extract wavelength range from spectrum',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Input spectrum')
    parser.add_argument('output', type=str, help='Output spectrum')
    parser.add_argument('start', type=float, help='Start of wavelength range')
    parser.add_argument('end', type=float, help='End of wavelength range')

    # argparse wrapper for create_grid
    parser.set_defaults(func=lambda args: extract_spectrum(**vars(args)))


def extract_spectrum(input: str, output: str, start: float, end: float, **kwargs):
    """Extracts the given wavelength range from input and store it as output.

    Args:
        input: Input filename
        output: Output filename
        start: Wavelength start
        end: Wavelength end
    """

    # load spectrum
    with FitsSpectrum(input) as fs_in:
        # open spectrum to write
        with FitsSpectrum(output, 'w') as fs_out:
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