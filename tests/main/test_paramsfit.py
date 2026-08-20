import numpy as np
import pytest
import scipy.linalg
from lmfit import Parameters

from spexxy.data import Spectrum
from spexxy.component import StarComponent
from spexxy.main.paramsfit import ParamsFit, Legendre


def _flux(wave, center, depth, sigma):
    """A flat continuum with a single Gaussian absorption dip, always > 0."""
    return 1.0 - depth * np.exp(-((wave - center) ** 2) / (2 * sigma ** 2))


class TestParamsFitWeightConvergenceConfig:
    def test_accepts_fast_and_full(self):
        assert ParamsFit(weight_convergence='fast')._weight_convergence == 'fast'
        assert ParamsFit(weight_convergence='full')._weight_convergence == 'full'

    def test_defaults_to_full(self):
        assert ParamsFit()._weight_convergence == 'full'

    def test_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            ParamsFit(weight_convergence='bogus')


class TestParamsFitSingleComponentWeight:
    """Characterizes the single-component branch of ParamsFit._get_model (paramsfit.py:557-565).

    This branch is NOT touched by the fix for GitHub issue #39 (only the
    len(cmps) > 1 branch is), so these tests pin its current behavior as a
    regression guard for the refactor.
    """

    def _setup(self, weight_value):
        wave = np.linspace(4500, 5500, 1500)
        tmpl_wave = np.linspace(4400, 5600, 1800)  # wider, avoids resample edge NaNs

        template = Spectrum(flux=_flux(tmpl_wave, 4900, 0.4, 40), wave=tmpl_wave)
        cmp = StarComponent(template, name="CMP")

        w_true = 2.7
        spec = Spectrum(flux=w_true * _flux(wave, 4900, 0.4, 40), wave=wave.copy())

        pf = ParamsFit(poly_degree=5)
        pf._cmps = [cmp]
        pf._spec = spec
        pf._valid = np.ones(len(wave), dtype=bool)
        pf._weight = np.full(len(wave), weight_value)
        pf._mult_poly = Legendre(spec, 5)

        model = pf._get_model(cmp.make_params())
        return w_true, cmp, model, spec

    def test_uniform_weight_one_reproduces_true_weight(self):
        """With no pixel weighting (the common case), the reported weight matches
        the true scale factor exactly, and the returned model matches the spectrum."""
        w_true, cmp, model, spec = self._setup(weight_value=1.0)

        assert cmp.weight == pytest.approx(w_true, rel=1e-3)
        assert model.flux == pytest.approx(spec.flux, abs=1e-3)

    def test_uniform_nonunity_weight_scales_reported_weight_by_its_square(self):
        """Known quirk of the existing formula (paramsfit.py:561-565): self._weight
        is applied asymmetrically (only to the spectrum side, not the model side), so
        a uniform pixel weight c biases the *reported* cmp.weight by c^2 -- even
        though the returned model still matches the spectrum (the final continuum
        renormalization step self-corrects the model shape, but not the reported
        weight value). This is a separate, pre-existing issue from #39; pinned here
        so it isn't silently changed by unrelated work.
        """
        c = 3.0
        w_true, cmp, model, spec = self._setup(weight_value=c)

        assert cmp.weight == pytest.approx(c ** 2 * w_true, rel=1e-3)
        assert model.flux == pytest.approx(spec.flux, abs=1e-3)


class TestParamsFitMultiComponentWeight:
    """Tests for ParamsFit._fit_component_weights (paramsfit.py:600-625), the function
    at the center of GitHub issue #39: it ignores self._mult_poly and self._weight,
    unlike the single-component branch right above it.
    """

    @pytest.fixture
    def two_components(self):
        wave = np.linspace(4500, 5500, 1500)
        m1 = Spectrum(flux=_flux(wave, 4800, 0.4, 15), wave=wave.copy())
        m2 = Spectrum(flux=_flux(wave, 5200, 0.35, 20), wave=wave.copy())
        return wave, m1, m2

    def test_ignores_continuum_and_pixel_weight_bug(self, two_components):
        """Demonstrates issue #39: with a non-flat continuum estimate and non-uniform
        pixel weights in play (both realistic mid-fit conditions), the current
        implementation's recovered weights are measurably biased away from the true
        values, because it fits models directly against the spectrum without folding
        in self._mult_poly.values or self._weight.

        EXPECTED TO FAIL against the current implementation -- documents the bug.
        Should pass once _fit_component_weights is fixed to include both terms,
        matching the single-component branch's convention.
        """
        wave, m1, m2 = two_components
        w1_true, w2_true = 1.4, 0.8

        # non-trivial continuum estimate, as if fit in a previous LM iteration
        poly_true = 1.0 + 0.15 * np.sin((wave - 4500) / 300.0)
        assert np.all(poly_true > 0)

        spec = Spectrum(flux=poly_true * (w1_true * m1.flux + w2_true * m2.flux), wave=wave.copy())

        pf = ParamsFit(poly_degree=5)
        cmp1, cmp2 = StarComponent(m1, name="A"), StarComponent(m2, name="B")
        pf._cmps = [cmp1, cmp2]
        pf._spec = spec
        pf._valid = np.ones(len(wave), dtype=bool)
        pf._weight = 1.0 + 0.5 * np.cos((wave - 4500) / 150.0)
        assert np.all(pf._weight > 0)
        pf._mult_poly = Legendre(spec, 5)
        pf._mult_poly.values = poly_true.copy()

        pf._fit_component_weights([m1, m2])

        assert cmp1.weight == pytest.approx(w1_true, rel=1e-3)
        assert cmp2.weight == pytest.approx(w2_true, rel=1e-3)

    def test_recovers_true_weights_when_continuum_is_flat_and_pixel_weight_uniform(self, two_components):
        """Sanity check: when the confounders from the test above are absent (flat
        continuum, uniform weight), even the current buggy implementation recovers
        the true weights exactly, since the omitted terms are then identity
        operations. Confirms the bias above is specifically due to the omission of
        mult_poly/weight, not the test setup or the linear solve itself.
        """
        wave, m1, m2 = two_components
        w1_true, w2_true = 1.4, 0.8

        spec = Spectrum(flux=w1_true * m1.flux + w2_true * m2.flux, wave=wave.copy())

        pf = ParamsFit(poly_degree=5)
        cmp1, cmp2 = StarComponent(m1, name="A"), StarComponent(m2, name="B")
        pf._cmps = [cmp1, cmp2]
        pf._spec = spec
        pf._valid = np.ones(len(wave), dtype=bool)
        pf._weight = np.ones(len(wave))
        pf._mult_poly = Legendre(spec, 5)
        pf._mult_poly.values = np.ones(len(wave))

        pf._fit_component_weights([m1, m2])

        assert cmp1.weight == pytest.approx(w1_true, rel=1e-6)
        assert cmp2.weight == pytest.approx(w2_true, rel=1e-6)

    def test_full_convergence_reaches_lower_residual_than_fast(self):
        """Exercises the alternation loop in _get_model (not just _fit_component_weights
        in isolation): starting from an unseeded continuum estimate (Legendre's default
        all-ones, as at the very first LM evaluation), 'fast' does a single weight/poly
        alternation while 'full' iterates to convergence.

        Note this checks fit *residual*, not recovered weight values: with a
        continuum polynomial degree high enough to be degenerate with the
        component weight ratio, converging the alternation finds a low-residual
        (w1, w2, poly) combination that fits the data well, but not necessarily
        the one matching the true generating weights -- that degeneracy is exactly
        why ULySS's own docs (uly_fit_lin.pro) describe this as fundamentally hard,
        not just slow.
        """
        wave = np.linspace(4500, 5500, 1500)
        tmpl_wave = np.linspace(4400, 5600, 1800)
        m1t = Spectrum(flux=_flux(tmpl_wave, 4800, 0.4, 15), wave=tmpl_wave.copy())
        m2t = Spectrum(flux=_flux(tmpl_wave, 5200, 0.35, 20), wave=tmpl_wave.copy())

        w1_true, w2_true = 1.4, 0.8
        poly_true = 1.0 + 0.15 * np.sin((wave - 4500) / 300.0)
        spec = Spectrum(
            flux=poly_true * (w1_true * _flux(wave, 4800, 0.4, 15) + w2_true * _flux(wave, 5200, 0.35, 20)),
            wave=wave.copy(),
        )

        residuals = {}
        for mode in ('fast', 'full'):
            pf = ParamsFit(poly_degree=5, weight_convergence=mode)
            cmp1, cmp2 = StarComponent(m1t, name="A"), StarComponent(m2t, name="B")
            pf._cmps = [cmp1, cmp2]
            pf._spec = spec
            pf._valid = np.ones(len(wave), dtype=bool)
            pf._weight = np.ones(len(wave))
            pf._mult_poly = Legendre(spec, 5)  # unseeded, starts at all-ones

            params = Parameters()
            params += cmp1.make_params()
            params += cmp2.make_params()

            model = pf._get_model(params)
            residuals[mode] = np.sum((model.flux[pf._valid] - spec.flux[pf._valid]) ** 2)

        assert residuals['full'] < 1e-3
        assert residuals['fast'] > 1e-3
        assert residuals['full'] < residuals['fast']

    def test_full_convergence_terminates_before_cap(self):
        """Regression guard for the convergence test: 'full' mode must actually converge
        (stop alternating) rather than always run to the 500-iteration cap. Counts the
        number of _fit_component_weights calls inside a single _get_model evaluation;
        the alternation converges in well under 50 iterations here, so a value anywhere
        near the 500 cap indicates the exit criterion has regressed.
        """
        wave = np.linspace(4500, 5500, 1500)
        tmpl_wave = np.linspace(4400, 5600, 1800)
        m1t = Spectrum(flux=_flux(tmpl_wave, 4800, 0.4, 15), wave=tmpl_wave.copy())
        m2t = Spectrum(flux=_flux(tmpl_wave, 5200, 0.35, 20), wave=tmpl_wave.copy())

        w1_true, w2_true = 1.4, 0.8
        poly_true = 1.0 + 0.15 * np.sin((wave - 4500) / 300.0)
        spec = Spectrum(
            flux=poly_true * (w1_true * _flux(wave, 4800, 0.4, 15) + w2_true * _flux(wave, 5200, 0.35, 20)),
            wave=wave.copy(),
        )

        pf = ParamsFit(poly_degree=5, weight_convergence='full')
        cmp1, cmp2 = StarComponent(m1t, name="A"), StarComponent(m2t, name="B")
        pf._cmps = [cmp1, cmp2]
        pf._spec = spec
        pf._valid = np.ones(len(wave), dtype=bool)
        pf._weight = np.ones(len(wave))
        pf._mult_poly = Legendre(spec, 5)  # unseeded, starts at all-ones

        calls = []
        orig = pf._fit_component_weights
        def counting(models):
            calls.append(1)
            return orig(models)
        pf._fit_component_weights = counting

        params = Parameters()
        params += cmp1.make_params()
        params += cmp2.make_params()
        pf._get_model(params)

        assert 1 < len(calls) < 50


class TestParamsFitNegativeWeights:
    """Characterizes weight non-negativity in _fit_component_weights.

    Component weight represents a physical flux fraction of a template in a
    composite spectrum, so it can't be negative in this model -- which is why
    _fit_component_weights uses a bounded (>= 0) solve, matching ULySS's
    BVLS-constrained approach. These tests use a case deliberately constructed
    so the true generating combination needs a negative weight (two
    correlated/near-collinear templates), to make the effect of the bound
    concrete and measurable rather than argued from first principles.
    """

    def _setup(self):
        wave = np.linspace(4500, 5500, 1500)
        # near-collinear: same shape, small center offset
        m1 = Spectrum(flux=_flux(wave, 4900, 0.4, 40), wave=wave.copy())
        m2 = Spectrum(flux=_flux(wave, 4950, 0.4, 40), wave=wave.copy())

        w1_true, w2_true = 1.5, -0.5
        spec = Spectrum(flux=w1_true * m1.flux + w2_true * m2.flux, wave=wave.copy())

        pf = ParamsFit(poly_degree=5)
        cmp1, cmp2 = StarComponent(m1, name="A"), StarComponent(m2, name="B")
        pf._cmps = [cmp1, cmp2]
        pf._spec = spec
        pf._valid = np.ones(len(wave), dtype=bool)
        pf._weight = np.ones(len(wave))
        pf._mult_poly = Legendre(spec, 5)
        pf._mult_poly.values = np.ones(len(wave))

        return pf, cmp1, cmp2, m1, m2, spec

    def test_bounded_solve_clips_negative_weight_to_zero(self):
        pf, cmp1, cmp2, m1, m2, spec = self._setup()

        pf._fit_component_weights([m1, m2])

        # component B's true weight was negative -- the bound clips it to 0,
        # and component A absorbs some of the difference instead
        assert cmp2.weight == pytest.approx(0.0, abs=1e-8)
        assert cmp1.weight > 0

    def test_bounded_solve_costs_fit_quality_vs_unconstrained(self):
        """Measures the price of enforcing non-negativity: how much worse the fit
        residual gets compared to what the (unphysical) unconstrained solve would
        achieve, for a case that genuinely wants a negative weight.
        """
        pf, cmp1, cmp2, m1, m2, spec = self._setup()

        pf._fit_component_weights([m1, m2])
        resid_bounded = spec.flux - (cmp1.weight * m1.flux + cmp2.weight * m2.flux)
        chi2_bounded = np.sum(resid_bounded ** 2)

        unconstrained = scipy.linalg.lstsq(np.column_stack([m1.flux, m2.flux]), spec.flux)[0]
        resid_unconstrained = spec.flux - (unconstrained[0] * m1.flux + unconstrained[1] * m2.flux)
        chi2_unconstrained = np.sum(resid_unconstrained ** 2)

        # unconstrained recovers the exact (noise-free) generating combination;
        # bounded cannot, since it truly needs a negative weight -- the gap here
        # is the measured cost of the non-negativity constraint for this case
        assert chi2_unconstrained == pytest.approx(0.0, abs=1e-6)
        assert chi2_bounded > 1.0
