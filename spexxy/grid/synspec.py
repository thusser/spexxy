import copy
import shutil
from tempfile import mkdtemp
from typing import List, Any, Tuple, Dict, Union
import pandas as pd
import os

from .grid import Grid, GridAxis
from ..data import Spectrum
from ..interpolator import Interpolator

ABUND_AGSS = [
    (1, 'H', 12., 2),
    (2, 'He', 10.93, 2),
    (3, 'Li', 3.26, 1),
    (4, 'Be', 1.38, 1),
    (5, 'B', 2.79, 1),
    (6, 'C', 8.43, 2),
    (7, 'N', 7.83, 2),
    (8, 'O', 8.69, 2),
    (9, 'F', 4.56, 1),
    (10, 'Ne', 7.93, 2),
    (11, 'Na', 6.24, 1),
    (12, 'Mg', 7.60, 2),
    (13, 'Al', 6.45, 2),
    (14, 'Si', 7.51, 2),
    (15, 'P', 5.41, 1),
    (16, 'S', 7.12, 2),
    (17, 'Cl', 5.50, 1),
    (18, 'Ar', 6.40, 1),
    (19, 'K', 5.08, 1),
    (20, 'Ca', 6.34, 1),
    (21, 'Sc', 3.15, 1),
    (22, 'Ti', 4.95, 1),
    (23, 'V', 3.93, 1),
    (24, 'Cr', 5.64, 1),
    (25, 'Mn', 5.43, 1),
    (26, 'Fe', 7.50, 2),
    (27, 'Co', 4.99, 1),
    (28, 'Ni', 6.22, 1),
    (29, 'Cu', 4.19, 1),
    (30, 'Zn', 4.56, 1),
    (31, 'Ga', 3.04, 1),
    (32, 'Ge', 3.65, 1),
    (33, 'As', 2.30, 1),
    (34, 'Se', 3.34, 1),
    (35, 'Br', 2.54, 1),
    (36, 'Kr', 3.25, 1),
    (37, 'Rb', 2.36, 1),
    (38, 'Sr', 2.87, 1),
    (39, 'Y', 2.21, 1),
    (40, 'Zr', 2.58, 1),
    (41, 'Nb', 1.46, 1),
    (42, 'Mo', 1.88, 1),
    (43, 'Tc', 0, 0),
    (44, 'Ru', 1.75, 1),
    (45, 'Rh', 1.06, 1),
    (46, 'Pd', 1.65, 1),
    (47, 'Ag', 1.20, 1),
    (48, 'Cd', 1.71, 1),
    (49, 'In', 0.76, 1),
    (50, 'Sn', 2.04, 1),
    (51, 'Sb', 1.01, 1),
    (52, 'Te', 2.18, 1),
    (53, 'I', 1.55, 1),
    (54, 'Xe', 2.24, 1),
    (55, 'Cs', 1.08, 1),
    (56, 'Ba', 2.18, 1),
    (57, 'La', 1.10, 1),
    (58, 'Ce', 1.58, 1),
    (59, 'Pr', 0.72, 1),
    (60, 'Nd', 1.42, 1),
    (61, 'Pm', 0, 0),
    (62, 'Sm', 0.96, 1),
    (63, 'Eu', 0.52, 1),
    (64, 'Gd', 1.07, 1),
    (65, 'Tb', 0.30, 1),
    (66, 'Dy', 1.10, 1),
    (67, 'Ho', 0.48, 1),
    (68, 'Er', 0.92, 1),
    (69, 'Tm', 0.10, 1),
    (70, 'Yb', 0.92, 1),
    (71, 'Lu', 0.10, 1),
    (72, 'Hf', 0.85, 1),
    (73, 'Ta', -0.12, 1),
    (74, 'W',  0.65, 1),
    (75, 'Re', 0.26, 1),
    (76, 'Os', 1.40, 1),
    (77, 'Ir', 1.38, 1),
    (78, 'Pt', 1.62, 1),
    (79, 'Au', 0.80, 1),
    (80, 'Hg', 1.17, 1),
    (81, 'Tl', 0.77, 1),
    (82, 'Pb', 2.04, 1),
    (83, 'Bi', 0.65, 1),
    (84, 'Po', 0, 0),
    (85, 'At', 0, 0),
    (86, 'Rn', 0, 0),
    (87, 'Fr', 0, 0),
    (88, 'Ra', 0, 0),
    (89, 'Ca', 0, 0),
    (90, 'Th', 0.06, 1),
    (91, 'Pa', 0, 0),
    (92, 'U', -0.54, 1)
]


FORT5_HEADER = """%.2f %.2f
 T  F              ! LTE,  LTGRAY
 'nstf'            ! name of file containing non-standard flags
*
* frequencies
*
 2000
*
* data for atoms   
*
 92                 ! NATOMS
* mode abn modpf
"""


FORT5_FOOTER = """*
* data for ions
*
*iat   iz   nlevs  ilast ilvlin  nonstd typion  filei
*
   1   -1     1      0      0      0    ' H- ' 'data/hmin.dat'
   1    0     9      0      0      0    ' H 1' 'data/h1.dat'
   1    1     1      1      0      0    ' H 2' ' '
   2    0    24      0      0      0    'He 1' 'data/he1.dat'
   2    1     1      1      0      0    'He 2' ' '
   6    0    40      0      0      0    ' C 1' 'data/c1.dat'
   6    1    22      0      0      0    ' C 2' 'data/c2.dat'
   6    2     1      1      0      0    ' C 3' ' '
   7    0    34      0      0      0    ' N 1' 'data/n1.dat'
   7    1    42      0      0      0    ' N 2' 'data/n2_32+10lev.dat'
   7    2     1      1      0      0    ' N 3' ' '
   8    0    33      0      0      0    ' O 1' 'data/o1_23+10lev.dat'
   8    1    48      0      0      0    ' O 2' 'data/o2_36+12lev.dat'
   8    2     1      1      0      0    ' O 3' ' '
  10    0    35      0      0      0    'Ne 1' 'data/ne1_23+12lev.dat'
  10    1    32      0      0      0    'Ne 2' 'data/ne2_23+9lev.dat'
  10    2     1      1      0      0    'Ne 3' ' '
  12    0    41      0      0      0    'Mg 1' 'data/mg1.t'
  12    1    25      0      0      0    'Mg 2' 'data/mg2.dat'
  12    2     1      1      0      0    'Mg 3' ' '
  13    0    10      0      0      0    'Al 1' 'data/al1.t'
  13    1    29      0      0      0    'Al 2' 'data/al2_20+9lev.dat'
  13    2     1      1      0      0    'Al 3' ' '
  14    0    45      0      0      0    'Si 1' 'data/si1.t'
  14    1    40      0      0      0    'Si 2' 'data/si2_36+4lev.dat'
  14    2     1      1      0      0    'Si 3' ' '
  16    0    41      0      0      0    ' S 1' 'data/s1.t'
  16    1    33      0      0      0    ' S 2' 'data/s2_23+10lev.dat'
  16    2     1      1      0      0    ' S 3' ' '
  26    0    30      0      0      0    'Fe 1' 'data/fe1.dat'
  26    1    36      0      0     -1    'Fe 2' 'data/fe2va.dat'
   0    0                                      'data/gf2601.gam'
                                               'data/gf2601.lin'
                                               'data/fe2p_14+11lev.rap'
  26    2    50      0      0     -1    'Fe 3' 'data/fe3va.dat'
   0    0                                      'data/gf2602.gam'
                                               'data/gf2602.lin'
                                               'data/fe3p_22+7lev.rap'
  26    3     1      1      0      0    'Fe 4' ' '
   0    0     0     -1      0      0    '    ' ' '
*
"""


FORT55 = """10   52   0        ! imode idstd iprin
0    0   0   1     ! inmod intrpl ichang ichemc
0    0   0   0   0 ! iophli nunalp nunbet nungam nunbal
1    0   0   0   0 ! ifreq inlte icontl inlist ifhe2
1    0   0         ! ihydpr ihe1pr ihe2pr
%4d  %4d 40  0  1.e-5 0.03 ! alam0 alast cutof0 cutofs relop space
1  26              ! nmlist, (iunitm(i),i=1,nmlist) for molecular linelists
"""


class SynspecGrid(Grid):
    """Synthesizes a new spectrum with Synspec at given grid positions."""

    def __init__(self, synspec: str, models: Grid, linelist: str, mollist: str, datadir: str, element: str,
                 range: Tuple[float, float], vturb: Union[str, float] = 2.0, *args, **kwargs):
        """Constructs a new Grid.

        Args:
            synspec: Full path to synspec exectuble
            models: Grid with model atmospheres
            linelist: File with line list
            mollist: File with molecular list
            datadir: Name of data directory
            element: Element to change
            range: Tuple of start/end wavelenghts
            vturb: Either the microturbulence or a CSV file containing a table
        """
        Grid.__init__(self, axes=None, *args, **kwargs)

        # store
        self._synspec = synspec
        self._linelist = linelist
        self._mollist = mollist
        self._datadir = datadir
        self._element = element
        self._range = range

        # load grid
        self._models: Grid = self.get_objects(models, [Grid, Interpolator], 'grids', self.log, single=True)

        # add and init axes
        self._axes = copy.deepcopy(self._models.axes())
        self._axes.append(GridAxis(name=element, initial=0.))

        # vturb
        if isinstance(vturb, int) or isinstance(vturb, float):
            self._vturb = float(vturb)
        elif isinstance(vturb, str):
            filename = os.path.expandvars(vturb)
            self._vturb = pd.read_csv(filename, index_col=['Teff', 'logg', '[M/H]', '[alpha/M]'], dtype=float)

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        return self._models.all()

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        return tuple(params[:-1]) in self._models

    def filename(self, params: Tuple) -> str:
        """Returns filename for given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Filename.
        """
        return None

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """

        # get params
        teff, logg, feh, alpha, el = params

        # find element in models grid
        mod = self._models.filename(params[:-1])

        # temp directory
        tmp = os.path.abspath(mkdtemp())

        # change path
        cwd = os.getcwd()
        os.chdir(tmp)

        # make sure to delete temp directory in the end
        try:
            # write fort.5 and file with non-standard flags
            self._write_fort5(teff, logg, feh, alpha)
            self._write_nstf(teff, logg, feh, alpha)

            # write config
            self._write_fort55(*self._range)

            # write element changes
            self._write_fort56({self._element: el})

            # create symlinks
            os.symlink(os.path.expandvars(self._synspec), 'synspec')
            os.symlink(mod, 'fort.8')
            os.symlink(os.path.expandvars(self._linelist), 'fort.19')
            os.symlink(os.path.expandvars(self._mollist), 'fort.26')
            os.symlink(os.path.expandvars(self._datadir), 'data')

            # run synspec
            os.system('./synspec < fort.5 > fort.6')

            # read output
            d = pd.read_csv('fort.7', delim_whitespace=True, names=['wave', 'flux'])

            # return spec
            spec = Spectrum(wave=d['wave'], flux=d['flux']).resample_const(step=0.02)
            return spec

        finally:
            # return to old directory and clean up
            os.chdir(cwd)
            shutil.rmtree(tmp)

    def _write_fort5(self, teff, logg, feh, alpha):
        with open('fort.5', 'w') as f:
            # write header
            f.write(FORT5_HEADER % (teff, logg))

            # write abundances
            for no, el, abund, mode in ABUND_AGSS:
                # calculate abundance
                if no == 0:
                    # H
                    a = 0
                elif no == 1:
                    # He
                    a = 10.**(abund - 12) if mode > 0 else 0.
                elif no in [8, 10, 12, 14, 16, 18, 20, 22]:
                    # alpha elements
                    a = 10. ** (abund - 12 + feh + alpha) if mode > 0 else 0.
                else:
                    # other
                    a = 10. ** (abund - 12 + feh) if mode > 0 else 0.

                # write it
                f.write('%d %.3g 0 !%s\n' % (mode, a, el))

            # write footer
            f.write(FORT5_FOOTER)

    def _write_nstf(self, teff, logg, feh, alpha_m):
        """Write file with non-standard flags."""

        # get vturb
        if isinstance(self._vturb, float):
            vturb = self._vturb
        elif isinstance(self._vturb, pd.DataFrame):
            vturb = float(self._vturb.loc[teff, logg, feh, alpha_m])
        else:
            return

        # write file
        with open('nstf', 'w') as f:
            f.write('ND=64,VTB=%.2f, IFMOL=1' % vturb)

    def _write_fort55(self, wstart, wend):
        with open('fort.55', 'w') as f:
            f.write(FORT55 % (wstart, wend))

    def _write_fort56(self, abunds: Dict[str, float]):
        """Write fort.56 file.

        To create the fort.56 that contains the updated abundance (abund_up) for the element wanted
        All we need is a file with :
            1
            iat abund_up

        Args:
            abunds: Dictionary as el->[X/H] abundance
        """

        with open('fort.56', 'w') as f:
            # write number of changes
            f.write('%d\n' % len(abunds))

            # write changes
            for user_el, user_abund in abunds.items():
                # find el in AGSS
                for no, el, abund, _ in ABUND_AGSS:
                    if el == user_el:
                        f.write('%d %.3g\n' % (no, 10. ** (abund - 12 + user_abund)))
                        break
                else:
                    raise ValueError('Element %s not found.' % user_el)


__all__ = ['SynspecGrid']
