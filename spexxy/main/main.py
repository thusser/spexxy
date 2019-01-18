from typing import List

from spexxy.object import spexxyObject


class MainRoutine(spexxyObject):
    """MainRoutine is the base class for all main routines."""

    def __init__(self,  *args, **kwargs):
        """Initialize a new MainRoutine object"""
        spexxyObject.__init__(self, *args, **kwargs)

    def parameters(self) -> List[str]:
        """Get list of parameters fitted by this routine.

        Returns:
            List of parameter names (including prefix) fitted by this routine.
        """
        return []

    def columns(self) -> List[str]:
        """Get list of columns returned by __call__.

        The returned list shoud include the list from parameters().

        Returns:
            List of columns returned by __call__.
        """

        # build list of columns from parameters
        columns = []
        for p in self.parameters():
            columns += [p.upper(), p.upper() + ' ERR']
        return columns

    def __call__(self, filename: str) -> List[float]:
        """Start the routine on the given file.

        Args:
            filename: Name of file to process.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """
        raise NotImplementedError


__all__ = ['MainRoutine']
