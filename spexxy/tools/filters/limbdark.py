import os

import scipy
from astropy.io import fits

from spexxy.data import Filter, Spectrum
from spexxy.grid import FilesGrid


# define list of FITS keywords to copy to new files
KEYWORDS = ['PHXTEFF', 'PHXLOGG', 'PHXM_H', 'PHXDUST', 'PHXXI_L', 'PHXXI_M', 'PHXXI_N', 'PHXMASS', 'PHXREFF', 'PHXLUM',
            'PHXMXLEN', 'PHXBUILD', 'PHXVER', 'DATE', 'PHXEOS']


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('limbdark', help='Apply filter to specific intensities file(s)')
    parser.set_defaults(func=run)

    # parameters
    parser.add_argument('input', help='Spectrum or grid file (in --grid mode)', type=str)
    parser.add_argument('output', help='Output filename or directory (in --grid mode)', type=str)
    parser.add_argument('filter', help='Filter name or filename', type=str)
    parser.add_argument('-g', '--grid', help='Process all files in grid file given as input', action='store_true')


def run(args):
    cld = ConvolveLimbDark(args.input, args.output, args.filter, args.grid)
    cld()


class ConvolveLimbDark:
    def __init__(self, input: str, output: str, fltr: str, grid_mode: bool = False):
        """Initialises object

        Args:
            input: Input filename/grid
            output: Output filename/directory
            fltr: Filter name
            grid_mode: Whether input is a grid
        """

        # store
        self.input = input
        self.output = output
        self.grid_mode = grid_mode

        # find filter
        self.filter = Filter(fltr)
        if self.filter.filename is None:
            raise ValueError('Could not find filter: ' + fltr)
        self.filter_resampled = False

    def __call__(self):
        """Process input file/grid"""

        # grid mode or single spectrum?
        if self.grid_mode:
            # load grid
            grid = FilesGrid(self.input)

            # process it
            self.process_grid(grid)

        else:
            # process single file
            self.process_single(self.input, self.output)

    def process_grid(self, grid):
        """Convolves all specific intensity spectra in the given grid.

        Args:
            grid: Grid to convolve.
        """

        # loop all entries in grid
        for params in grid.all():
            # get input filename
            in_filename = grid.filename(params)

            # create output filename
            out_filename = os.path.join(self.output, grid.filename(params, absolute=False))

            # process file
            self.process_single(in_filename, out_filename)

    def process_single(self, input, output):
        """Convolves a specific intensities spectrum with the given filter and stores result."""

        # load data and mus
        mu = fits.getdata(input, 'MU')
        data, hdr = fits.getdata(input, header=True)
        wave_start = hdr['CRVAL1']
        wave_step = hdr['CDELT1']

        # loop all mus
        intensities = []
        for k in range(len(mu)):
            # create spectrum
            spec = Spectrum(flux=data[k, :], wave_start=wave_start, wave_step=wave_step)

            # resample filter?
            if not self.filter_resampled:
                self.filter.resample(spec=spec, fill_value=0.)
                self.filter_resampled = True

            # apply filter
            intensity = self.filter.integrate(spec)
            intensities.append(intensity)

        # normalize to 1 at centre
        # intensities /= intensities[-1]

        # integrate I(mu)*mu
        H = scipy.integrate.trapz(intensities * mu, mu)

        # scale limb darkening as I(mu)/2H
        intensities /= 2 * H

        # create HDU
        hdu_primary = fits.PrimaryHDU(intensities)
        hdu_primary.header['WAVE'] = 'MU'

        # set headers
        for key in KEYWORDS:
            hdu_primary.header[key] = hdr[key]

        # HDU for MUs
        hdu_mu = fits.ImageHDU(mu)
        hdu_mu.name = 'MU'

        # path exists?
        path = os.path.dirname(output)
        if not os.path.exists(path):
            os.makedirs(path)

        # store it
        fits.HDUList([hdu_primary, hdu_mu]).writeto(output, overwrite=True)
