import yaml

from .init import Init
from ..component import Component


class InitFromYAML(Init):
    """Initializes a component from values in a given YAML file.

    This class, when called, initializes the parameters of a given component to the values provided in a YAML file.
    """

    def __init__(self, filename: str, *args, **kwargs):
        """Initializes a new Init object.

        Args:
            filename: Name of YAML file to read values from.
        """
        Init.__init__(self, *args, **kwargs)
        self._filename = filename

    def __call__(self, cmp: Component, filename: str):
        """Initializes parameters of the given component with values from the configuration

        Args:
            cmp: Component to initialize.
            filename: Unused.
        """

        # load YAML file
        with open(self._filename, 'r') as f:
            values = yaml.safe_load(f)

        # loop initial values
        for key, val in values.items():
            # does parameter exist in component?
            if key in cmp.param_names:
                # set it
                self.log.info('Setting initial value for "%s" of component "%s" to %f...', key, cmp.prefix, val)
                cmp[key] = val


__all__ = ['InitFromYAML']
