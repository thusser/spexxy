import sys
import lmfit
import numpy as np
import scipy.linalg
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from typing import List

from spexxy.data import FitsSpectrum, Spectrum
from spexxy.component import Component
from spexxy.mask import Mask
from spexxy.weight import Weight
from spexxy.data import SpectrumFitsHDU
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

    def __call__(self, filename: str) -> List[float]:
        """Start the routine on the given file.

        Args:
            filename: Name of file to process.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """
        raise NotImplementedError


__all__ = ['MainRoutine']
