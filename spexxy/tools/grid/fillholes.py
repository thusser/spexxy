import logging
import os
from typing import Callable
import numpy as np


log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('fillholes', help='Fill holes in a file grid')
    parser.add_argument('grid', type=str, help='Name of grid CSV', default='grid.csv')
    parser.add_argument('output', type=str, help='Directory to store interpolated spectra in', default='interpolated/')

    # argparse wrapper for fill_holes
    def run(args):
        fill_holes = FillGridHoles(**vars(args))
        fill_holes()
    parser.set_defaults(func=run)


class FillGridHoles:
    def __init__(self, grid, output, **kwargs):
        from ...grid import FilesGrid

        # load grid
        self.grid_csv = grid
        self.grid = FilesGrid(grid)
        self.output = output

        # create directory?
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def __call__(self):
        # find holes and call callback
        self.find_missing(callback=self._fill_holes_callback)

    @staticmethod
    def _subst_params(params: list, axis: int, value: float) -> tuple:
        """Convenience method that takes a list of parameters and substitutes the one on the given axis with the
        given value.

        Args:
            params: List of parameters, one per axis.
            axis: Number of the axis to change the value for, i.e. the index in the params list.
            value: New value for axis.

        Returns:
            New params tuple.
        """
        copy = list(params)
        copy[axis] = value
        return tuple(copy)

    def find_missing(self, axis: int = None, params: list = None, callback: Callable = None) -> list:
        """Finds missing items in the grid.

        Args:
            axis: Current axis to work on - if None, starts with last one.
            params: Set of (incomplete) parameters to work on.
            callback: Method that gets called with missing items for combinations of all axes except the first one.

        Returns:
            List of tuples of missing items in grid.
        """

        # we start with the last axis
        if axis is None:
            axis = self.grid.num_axes() - 1

        # init missing and params
        missing = []
        if params is None:
            params = []

        # are we at last remaining axis already?
        if axis == 0:
            # get first and last valid value on this axis
            first = None
            last = None
            for v in self.grid.axis_values(axis):
                if tuple([v] + params) in self.grid:
                    if first is None:
                        first = v
                    last = v

            # found any?
            if first is not None and last is not None and first != last:
                # now find missing
                missing_first_axis = []
                for v in self.grid.axis_values(axis):
                    if first < v < last and not tuple([v] + params) in self.grid:
                        missing += [tuple([v] + params)]
                        missing_first_axis += [v]

                # callback?
                if len(missing_first_axis) > 0 and callback is not None:
                    callback(params, missing_first_axis)

        else:
            # if it's not the last remaining axis, we just go deeper into the recursion
            for v in self.grid.axis_values(axis):
                missing += self.find_missing(axis - 1, [v] + params, callback=callback)

        # return list of missing items in grid
        return missing

    def _fill_holes_callback(self, params, values):
        from scipy.interpolate import interp1d
        from ...data import SpectrumFits

        # log
        log.info('Found %d missing spectra in (x,%s).', len(values), (','.join(str(p) for p in params)))

        # load all available spectra on this axis
        log.info('Loading existing spectra...')
        input_values = [v for v in self.grid.axis_values(0) if tuple([v] + params) in self.grid]
        spectra = [self.grid(tuple([v] + params)) for v in input_values]
        log.info('Found %d existing spectra.', len(spectra))

        # get fluxes
        input_fluxes = np.empty((len(spectra), len(spectra[0])))
        for i, spec, in enumerate(spectra):
            input_fluxes[i, :] = spec.flux

        # do interpolation by looping all wavelength points
        log.info('Interpolate on first axis at values %s.', (', '.join([str(v) for v in values])))
        output_fluxes = np.empty((len(values), len(spectra[0])))
        for w in range(output_fluxes.shape[1]):
            ip = interp1d(input_values, input_fluxes[:, w], kind='linear')
            output_fluxes[:, w] = ip(values)

        # create output spectra, copy fluxes, and save them
        for i in range(output_fluxes.shape[0]):
            # create spectrum and copy flux
            spec = SpectrumFits(spec=spectra[0], copy_flux=False)
            spec.flux = output_fluxes[i, :]

            # create filename
            filename = os.path.join(self.output, '_'.join('%.1f' % f for f in [values[i]] + params) + '.fits')

            # write file
            log.info('Writing spectrum to %s...', filename)
            spec.save(filename)

            # add to grid
            log.info('Adding spectrum to CSV file...')
            with open(self.grid_csv, 'a') as csv:
                csv.write('%s,%s\n' % (filename, ','.join(str(f) for f in [values[i]] + params)))
