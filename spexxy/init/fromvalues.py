from .init import Init
from ..component import Component


class InitFromValues(Init):
    """Initializes a component from given values.

    This class, when called, initializes the parameters of a given component to the provided values from a dict.
    """

    def __init__(self, values: dict, *args, **kwargs):
        """Initializes a new Init object.

        Args:
            values: Dictionary of key/value pairs defining initial values for the component's parameters.
        """
        Init.__init__(self, *args, **kwargs)
        self._values = values

    def __call__(self, cmp: Component, filename: str):
        """Initializes parameters of the given component with values from the configuration

        Args:
            cmp: Component to initialize.
            filename: Unused.
        """

        # loop initial values
        for key, val in self._values.items():
            # does parameter exist in component?
            if key in cmp.param_names:
                # set it
                self.log.info('Setting initial value for "%s" of component "%s" to %f...', key, cmp.prefix, val)
                cmp[key] = val


__all__ = ['InitFromValues']
