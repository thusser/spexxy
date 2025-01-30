import os
from enum import Enum
from typing import Callable, Union

from .component import Component
from ..data import LOSVD, Vsini, Spectrum, LSF
from ..object import create_object


class Broadening(Enum):
    LOSVD = 1
    VSINI = 2


class SpectrumComponent(Component):
    """SpectrumComponent is the base Component class for all components that deal with spectra."""

    def __init__(self, name: str, broadening_type: Broadening = Broadening.LOSVD, losvd_hermite: bool = False,
                 vac_to_air: bool = False, lsf: Union[LSF, str, dict] = None, *args, **kwargs):
        """Initializes a new SpectrumComponent.

        Args:
            name: Name of new component
            broadening_type: Type of broadening (LOSVD, VSINI)
            losvd_hermite: Whether Hermite polynomials should be used for the LOSVD
            vac_to_air: If True, vac_to_air() is called on spectra returned from the model_func
            lsf: LSF to apply to spectrum
        """
        Component.__init__(self, name, *args, **kwargs)
        self._vac_to_air = vac_to_air
        if isinstance(broadening_type, int):
            self._broadening_type = Broadening(broadening_type)
        elif isinstance(broadening_type, str):
            self._broadening_type = Broadening[broadening_type]
        else:
            self._broadening_type = broadening_type
        self._losvd_hermite = losvd_hermite

        # add losvd parameters
        self.set('v', min=-2000., max=2000., value=1.)
        if self._broadening_type == Broadening.LOSVD:
            self.set('sig', min=0., max=500., value=10.)
            if self._losvd_hermite:
                self.set('h3', min=-0.3, max=0.3, value=0.)
                self.set('h4', min=-0.3, max=0.3, value=0.)
                self.set('h5', min=-0.3, max=0.3, value=0.)
                self.set('h6', min=-0.3, max=0.3, value=0.)
        elif self._broadening_type == Broadening.VSINI:
            self.set('vsini', min=0, max=800., value=20.)
            self.set('epsilon', min=0., max=1., value=0.5)

        # get lsf
        if isinstance(lsf, LSF):
            self._lsf = lsf
        elif isinstance(lsf, str):
            self._lsf = LSF.load(lsf)
        elif isinstance(lsf, dict):
            self._lsf = create_object(lsf)
        else:
            self._lsf = None

    def _model_func(self):
        """Return actual model."""
        raise NotImplementedError

    def __call__(self, **kwargs):
        """Model function that creates a spectrum with the given parameters and shift/convolve it using the given LOSVD.

        Args:
            kwargs: Values to overwrite

        Returns:
            The model from the component
        """

        # overwrite any?
        for key, val in kwargs.items():
            if key in self.parameters:
                self.set(key, value=val)

        # get LOSVD parameters
        if self._broadening_type == Broadening.LOSVD:
            losvd_params = ['v', 'sig'] + (['h3', 'h4', 'h5', 'h6'] if self._losvd_hermite else [])
        elif self._broadening_type == Broadening.VSINI:
            losvd_params = ['v', 'vsini', 'epsilon']
        else:
            raise ValueError('Unknown broadening type: ', self._broadening_type)
        losvd = [self[p] for p in losvd_params]

        # get  model
        model = self._model_func()

        # apply LSF
        if self._lsf is not None:
            self._lsf.resample(model)
            model = self._lsf(model)

        # apply losvd
        self._apply_losvd(model, losvd, self._broadening_type)

        # vac2air
        if self._vac_to_air:
            model.vac_to_air()

        # return result
        return model

    @staticmethod
    def _apply_losvd(model, losvd, broadening_type=Broadening.LOSVD):
        """Apply LOSVD with the given parameters to the given model.

        WARNING: We will NOT FIT line broadening, if model spectra are in LAMBDA mode!

        Args:
            model: Model to apply LOSVD to.
            losvd: LOSVD parameters, required number depends on value of 'broadening'
                   Gauss-Hermite (LOSVD): (v, sig, <h3, h4, h5, h6>)
                   Rotational (VSINI): (v, vsini, epsilon)
            broadening_type: Broadening method to use.
        """
        if losvd[1] < 1e-5 or model.wave_mode == Spectrum.Mode.LAMBDA:
            # in LAMBDA mode, no LOSVD is supported
            model.redshift(losvd[0])
        else:
            # full LOSVD for sig/vsini>0 and LOG mode
            if model.wave_step == 0:
                # resample to const, if necessary
                model = model.resample_const()
            # initialize LOSVD/Vsini application
            if broadening_type == Broadening.LOSVD:
                kernel = LOSVD(losvd)
            elif broadening_type == Broadening.VSINI:
                kernel = Vsini(losvd)
            # apply it
            model.flux = kernel(model)


__all__ = ['SpectrumComponent']
