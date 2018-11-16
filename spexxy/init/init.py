from ..object import spexxyObject
from ..component import Component


class Init(spexxyObject):
    """Init is the base class for all objects that can initialize values of a component."""

    def __init__(self, *args, **kwargs):
        """Initialize an Init object."""
        spexxyObject.__init__(self, *args, **kwargs)

    def __call__(self, cmp: Component, filename: str):
        """Initializes values for the given component.

        Parameters
        ----------
        cmp : Component
            Component to initialize.
        filename : str
            Name of file containing spectrum to create mask for.
        """
        raise NotImplementedError
