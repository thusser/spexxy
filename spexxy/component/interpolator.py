from .spectrumcomponent import SpectrumComponent
from ..interpolator import Interpolator


class InterpolatorComponent(SpectrumComponent):
    """An Interpolator component takes an interpolator and adds LOSVD parameters."""

    def __init__(self, interpolator: Interpolator, name: str = "STAR", *args, **kwargs):
        """Initializes a new Grid component.

        Parameters
        ----------
        interpolator : Interpolator
            The interpolator to use for the component
        name : str
            Name of the component
        """
        SpectrumComponent.__init__(self, name, self._model_func, *args, **kwargs)
        self.log.info('Initializing grid component "%s"...', name)

        # get interpolator
        self._interpolator = self.get_objects(interpolator, Interpolator, 'interpolators')
        if isinstance(self._interpolator, list):
            self._interpolator = self._interpolator[0]

        # add parameters of interpolator
        for a in self._interpolator.axes():
            self.log.info('Found parameter %s with min=%.2f, max=%.2f, and initial=%.2f.',
                          a.name, a.min, a.max, a.initial)
            self.set(a.name, min=a.min, max=a.max, value=a.initial)

    def _model_func(self):
        """Get spectrum with given parameters.

        Returns
        -------
        Any
            Result from interpolator
        """

        # get values as tuple
        values = tuple([self[a.name] for a in self._interpolator.axes()])

        # interpolate
        return self._interpolator(values)


__all__ = ['InterpolatorComponent']
