import shutil
from tempfile import mkdtemp
from typing import List, Any, Tuple
import pandas as pd
import os

from .grid import Grid, GridAxis
from ..data import Spectrum


class SynspecGrid(Grid):
    """Synthesizes a new spectrum with Synspec at given grid positions."""

    def __init__(self, models: Grid, vturbs: Grid, synspec: str, inpmol: str, linelist: str, mollist: str, iat: float,
                 el_sol: float, datadir: str, gfmol: str, nstmol: str, inputfiles: str, *args, **kwargs):
        """Constructs a new Grid.

        Args:
            models: Grid with model atmospheres
            vturbs: Grid with micro turbulences
            synspec: Full path to synspec exectuble
            inpmol: Standard input for synspec
            linelist: File with line list
            mollist: File with molecular list
            iat: bla
            el_sol: bla
            datadir: Name of data directory
            gfmol: bla
            nstmol: bla
            inputfiles: bla
        """
        Grid.__init__(self, axes=None, *args, **kwargs)

        # store
        self._synspec = synspec
        self._inpmol = inpmol
        self._linelist = linelist
        self._mollist = mollist
        self._iat = iat
        self._el_sol = el_sol
        self._datadir = datadir
        self._gfmol = gfmol
        self._nstmol = nstmol
        self._inputfiles = inputfiles

        # load grids
        self._models: Grid = self.get_objects(models, Grid, 'grids', self.log, single=True)
        self._vturbs: Grid = self.get_objects(vturbs, Grid, 'grids', self.log, single=True)

        # init grid
        self._axes = self._models.axes()


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
        print(params)
        # get params
        teff, logg, feh, alpha, el = params

        # find element in models grid
        model_params = self._models.nearest(params[:-1])
        mod = self._models.filename(model_params)
        print(model_params, mod)

        # temp directory
        tmp = os.path.abspath(mkdtemp(dir='tmp'))

        # init
        subgrid = 'Z-1.0'
        self._make_inp5(tmp, teff, logg, subgrid)
        self._make_nst(tmp)

        self._make_inp56(tmp, self._iat, self._el_sol, el)
        self._copy_inputs(tmp, mod, self._inpmol, self._linelist, self._mollist)

        os.symlink(self._synspec, os.path.join(tmp, 'synspec'))

        cwd = os.getcwd()
        os.chdir(tmp)
        os.system('pwd')
        os.system('./synspec < fort.5 > fort.6')
        d = pd.read_csv('fort.7', delim_whitespace=True, names=['wave', 'flux'])
        os.chdir(cwd)

        spec = Spectrum(wave=d['wave'], flux=d['flux'])
        print(spec.wave_start, spec.wave_step)

        shutil.rmtree(tmp)
        return spec

    def _make_inp5(self, tmp, teff, logg, subgrid):
        ## read a template for the inp.5 file
        ## and replace teff and grav by the appropriate values

        with open(os.path.join(self._inputfiles, 'inp.5' + subgrid.strip('/')), 'r') as f:
            inp5temp = f.read()

        inp5 = inp5temp.replace('$TEFF', str(teff)).replace('$GRAV', str(logg))

        with open(os.path.join(tmp, 'fort.5'), 'w') as f:
            f.write(inp5)

    def _make_nst(self, tmp):
        ## I go open and read the original phoenix spectra, to retrieve information on the turbulent
        ## velocity of the original model atmosphere, which is contained in the header
        #pathf = 'lte04800-1.50-1.0.PHOENIX-ACES-AGSS-COND-2011-HiResMuse.fits'
        #hdu = fits.open(pathf)
        #vtb_phoenix = hdu[0].header['PHXXI_L']

        ## I will update the template nst file and put the right VTB value in there

        with open(self._nstmol, 'r') as f:
            temp = f.read()

        #nstnew = temp.replace('$VTB', str(vtb_phoenix))
        nstnew = temp.replace('$VTB', str(2))

        with open(os.path.join(tmp, 'nstphoenix'), 'w') as f:
            f.write(nstnew)

    def _make_inp56(self, tmp, iat, ab_sol, abund_up):
        # To create the fort.56 that contains the updated
        # abundance (abund_up) for the element wanted
        # All we need is a file with :
        # -----
        # 1
        # iat abund_up

        abund_elem = ab_sol * 10 ** abund_up

        with open(os.path.join(tmp, "fort.56"), 'w') as f:
            f.write('1 \n')
            f.write('%i  %0.3e \n' % (iat, abund_elem))

    def _copy_inputs(self, tmp, mod, inp55, llist, listmol):
        shutil.copy(mod, os.path.join(tmp, 'fort.8'))  # model atmosphere into fort.8
        shutil.copy(inp55, os.path.join(tmp, 'fort.55'))  # the standard input for synspec

        os.symlink(os.path.abspath(llist), os.path.join(tmp, 'fort.19'))
        os.symlink(os.path.abspath(listmol), os.path.join(tmp, 'fort.26'))
        os.symlink(self._gfmol, os.path.join(tmp, 'fort.27'))
        os.symlink(self._datadir, os.path.join(tmp, 'data'))


__all__ = ['SynspecGrid']
