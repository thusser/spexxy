import numpy as np

from ..testdata import data_filename
from spexxy.data import Filter, Spectrum, SpectrumAscii


class TestFilter(object):
    def test_load(self):
        """Test loading a filter."""

        # load it via Filter
        print(data_filename('filters'))
        f = Filter('V', path=data_filename('filters'))

        # load it as Spectrum
        s = SpectrumAscii(data_filename('filters/V.txt'), separator=None)

        # should be identical
        assert np.array_equal(f.wave, s.wave)
        assert np.array_equal(f.throughput, s.flux)

    def test_resample(self):
        """Test resampling a filter."""

        # load it via Filter
        f = Filter('V', path=data_filename('filters'))

        # integrate it
        int1 = np.trapz(f.throughput, f.wave)

        # load spectrum and resample filter
        s = Spectrum.load(data_filename('spectra/ngc6397id000010554jd2456865p5826f000.fits'))
        f.resample(spec=s, inplace=True)

        # integrate it again
        int2 = np.trapz(f.throughput, f.wave)

        # integral should not have changed significantly
        assert abs(int1 - int2) < 1

    def test_apply(self):
        """Test applying a filter."""

        # load it via Filter
        f = Filter('V', path=data_filename('filters'))

        # load spectrum, fix flux (which is given in 1e20erg/s/cm/cm2) and resample filter
        s = Spectrum.load(data_filename('spectra/ngc6397id000010554jd2456865p5826f000.fits'))
        s /= 1e20
        f.resample(spec=s, inplace=True)

        # calculate V mag as ST and VEGAMAG
        st = f.stmag(s)
        vega = f.vegamag(s)

        # F606W mag is 13.17, V mag should be close to 13.4
        assert abs(st - 13.4) < 0.1
        assert abs(vega - 13.4) < 0.1

    def test_list(self):
        """Test listing filters."""

        # list them
        filters = Filter.list('*', path=data_filename('filters'))

        # list should contain V and R
        names = sorted([f.filter_name for f in filters])
        assert ['R', 'V'] == names
