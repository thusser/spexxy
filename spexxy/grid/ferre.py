import copy
import numpy as np
from typing import Tuple, Any, List

from .grid import Grid, GridAxis
from ..data import Spectrum


class FerreGrid(Grid):
    def __init__(self, filename: str):
        """Load a Ferre grid.
        See: https://github.com/callendeprieto/ferre/blob/master/docs/ferre.pdf

        Args:
            filename: Name of file to load.
        """

        # init
        self._filename = filename
        self._n_of_dim = 0
        self._n_p = []
        self._labels = []
        self._llimits = []
        self._steps = []
        self._npix = None
        self._wave = []
        self._logw = None
        self._spec_start = None
        self._line_length = None
        self._line_pos = []
        self._params_line = {}

        # load grid
        self._load_grid()
        self._map_lines()

        # build axes
        axes = []
        for i in range(self._n_of_dim):
            values = [self._llimits[i] + k * self._steps[i] for k in range(self._n_p[i])]
            axes.append(GridAxis(name=self._labels[i],
                                 values=values,
                                 min=values[0],
                                 max=values[-1]))

        # init grid
        Grid.__init__(self, axes)

    def _load_grid(self):
        # open file
        with open(self._filename) as f:
            # check filetype
            if f.readline().strip() != '&SYNTH':
                raise ValueError('Given file is not a Ferre grid.')

            # read header
            for line in f:
                # end of header?
                if line.strip() == '/':
                    break

                # split by =
                p = line.index('=')
                key = line[:p].strip()
                value = line[p + 1:].strip()

                # remove quotes
                if len(value) > 0 and value[0] in ['"', "'"]:
                    value = value[1:-1]

                # set values
                if key == 'N_OF_DIM':
                    self._n_of_dim = int(value)
                    self._labels = [''] * self._n_of_dim
                    self._llimits = [0] * self._n_of_dim
                    self._steps = [0] * self._n_of_dim
                elif key == 'N_P':
                    self._n_p = [int(n) for n in value.split()]
                elif key.startswith('LABEL'):
                    i = int(line[line.index('(') + 1:line.index(')')])
                    self._labels[i - 1] = value
                elif key == 'LLIMITS':
                    self._llimits = [float(f) for f in value.split()]
                elif key == 'STEPS':
                    self._steps = [float(f) for f in value.split()]
                elif key == 'NPIX':
                    self._npix = int(value)
                elif key == 'WAVE':
                    self._wave = [float(f) for f in value.split()]
                elif key == 'LOGW':
                    self._logw = int(value) == 1

        # since we cannot do tell() on the file after looping lines, we reopen it now
        with open(self._filename) as f:
            # search for line that indicates end of header
            while f.readline().strip() != '/':
                pass

            # store start pos
            self._spec_start = f.tell()

            # determine line length, check first 100 lines
            self._line_length = len(f.readline())
            for i in range(100):
                if len(f.readline()) != self._line_length:
                    raise ValueError('Length of lines changes.')

    def _map_lines(self):
        """Map parameters to lines in file."""

        # init params to lower limits
        params = copy.copy(self._llimits)

        # map it
        self._map_line_axis(params)

    def _map_line_axis(self, params: list, line: int = 0, axis: int = 0) -> int:
        # loop axis values
        for i in range(self._n_p[axis]):
            # get value
            val = self._llimits[axis] + i * self._steps[axis]

            # set it
            p = copy.copy(params)
            p[axis] = val

            # go deeper?
            if axis == self._n_of_dim - 1:
                # no, read lines
                self._params_line[tuple(p)] = line
                line += 1
            else:
                # go deeper
                line = self._map_line_axis(p, line, axis + 1)

        # return new line pos
        return line

    def all(self) -> List[Tuple]:
        pass

    def __contains__(self, params: Tuple) -> bool:
        pass

    def __call__(self, params: Tuple) -> Any:
        # get line in file
        line = self._params_line[params]

        # get pos in file
        pos = self._spec_start + line * self._line_length

        # open file
        with open(self._filename) as f:
            # go to position of spectrum
            f.seek(pos)

            # read line
            flux = np.array([float(v) for v in f.readline().strip().split()])

        # return spectrum
        return Spectrum(flux=flux, wave_start=self._wave[0], wave_step=self._wave[1], wave_count=self._npix,
                        wave_mode=Spectrum.Mode.LOG10LAMBDA if self._logw else Spectrum.Mode.LAMBDA)


__all__ = ['FerreGrid']
