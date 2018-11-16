from .grid import GridComponent
from ..interpolator import Interpolator


class TelluricsComponent(GridComponent):
    """A Component for serving tellurics spectra from a given interpolator."""

    def __init__(self, interpolator: Interpolator, name: str = "TELLURICS", *args, **kwargs):
        """Initializes a new Grid component.

        Parameters
        ----------
        interpolator : Interpolator
            The interpolator to use for the component
        name : str
            Name of the component
        """
        GridComponent.__init__(self, interpolator, name, vac_to_air=True, *args, **kwargs)


__all__ = ['TelluricsComponent']
