from typing import Union

from .spectrumcomponent import SpectrumComponent
from ..data import Spectrum


class StarComponent(SpectrumComponent):
    def __init__(self, spec: Union[Spectrum, str], name: str = "STAR", *args, **kwargs):
        """Initializes a new Star component that just serves a single given spectrum.

        Parameters
        ----------
        spec : SpectrumComponent | str
            A Spectrum object or the filename of a spectrum to load
        name : str
            Name of new component
        """
        SpectrumComponent.__init__(self, name, *args, **kwargs)

        # load or copy spectrum?
        if isinstance(spec, Spectrum):
            self._spectrum = spec.copy(copy_flux=True)
        elif isinstance(spec, str):
            self._spectrum = Spectrum.load(spec)
        else:
            raise ValueError('Unknown type for spec.')

    def _model_func(self):
        """Get spectrum.

        Returns
        -------
        SpectrumComponent
            The spectrum that the component has been initialized with
        """
        return self._spectrum.copy()


__all__ = ['StarComponent']
