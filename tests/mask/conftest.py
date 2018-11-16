import pytest
import numpy as np

from spexxy.data import Spectrum


@pytest.fixture()
def spectrum():
    wave = np.arange(1, 10, 1)
    flux = np.random.uniform(0, 1, len(wave))
    yield Spectrum(flux=flux, wave=wave, wave_mode=Spectrum.Mode.LAMBDA)
