import pytest
import numpy as np

from spexxy.data import Spectrum


@pytest.fixture()
def spectrum():
    wave = np.arange(1., 10., 1.)
    flux = np.arange(1001., 1010., 1.)
    yield Spectrum(flux=flux, wave=wave, wave_mode=Spectrum.Mode.LAMBDA)
