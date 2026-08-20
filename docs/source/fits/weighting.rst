.. _weighting:

Component weights and the continuum polynomial
==============================================

When :class:`ParamsFit <spexxy.main.ParamsFit>` combines more than one component, it builds the
model spectrum as::

    model(λ) = P(λ) · Σ_k w_k · m_k(λ)

where ``m_k(λ)`` is the (resampled) model of component *k*, ``w_k`` is that component's weight, and
``P(λ)`` is a smooth multiplicative "continuum" polynomial of degree ``poly_degree`` (Legendre basis),
which absorbs any smooth residual difference between the sum of the components and the data (flux
calibration, reddening, etc.).

Both ``w`` and ``P`` are linear unknowns, but they are **multiplicative with each other**
(``P · w``), so the combined problem is *bilinear*, not linear: fitting the weights needs an estimate
of the polynomial and vice versa. A single pass is therefore not, in general, the correct solution.

The single-component case
-------------------------

With exactly one component the bilinearity disappears: the weight is the standard (weighted)
least-squares solution and is computed in closed form. No iteration is needed.

The multi-component case
------------------------

With two or more components, :class:`ParamsFit <spexxy.main.ParamsFit>` alternates between the two
half-problems until they are consistent (this mirrors ULySS's ``uly_fit_lin.pro``, which faces the
same bilinearity):

1. fix ``P`` and solve for the weights ``w``,
2. fix ``w`` and refit the polynomial ``P``,

repeating until convergence. The behavior is controlled by the ``weight_convergence`` option of
:class:`ParamsFit <spexxy.main.ParamsFit>`:

``weight_convergence = 'fast'``
    Performs a single alternation per model evaluation (like ULySS's ``MODECVG=0``). Fast, but the
    weights can occasionally be off because they depend on the polynomial left over from the previous
    evaluation.

``weight_convergence = 'full'`` (default)
    Alternates until convergence (like ULySS's ``MODECVG=2``), up to a hard cap of 500 iterations.
    Slower, but it removes the noisy, evaluation-dependent objective function that can stall the outer
    Levenberg-Marquardt optimization.

The option is ignored for single-component fits, which do not iterate.

.. _weighting_tol:

Convergence tolerance
---------------------

Convergence of the alternation is measured by the *relative change of the combined model*
(``Σ_k w_k · m_k · P``) between two successive alternations, and stops once that change drops below
``tol = 1e-4``.

Three properties of this test matter:

**It is relative, not absolute.** The component weights scale with the flux scale of the spectrum,
which differs by many orders of magnitude between inputs (for example ``~1e-14`` for
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}}` spectra versus ``~1`` for normalized spectra). An
absolute tolerance would mean different things for different data.

**It gates on the combined model, not the raw weights.** The weights and the polynomial are
degenerate: flux can be redistributed between ``w`` and ``P`` without changing the model at all.
Comparing only the weights would keep iterating long after the model — the quantity the fit actually
cares about — has settled.

**The value ``1e-4`` is deliberate and physically motivated.** The alternation converges only
*linearly* (a rate of roughly ``0.98`` per iteration for nearly-degenerate components, e.g. two stars
of similar shape), so the number of iterations needed grows rapidly as the tolerance is tightened.
But the fit residual is already at the data's noise floor after one or two iterations. A tolerance of
``1e-4`` is about two orders of magnitude below typical noise (``~1e-2`` at S/N ≈ 100), so the loop
stops after a couple of iterations with **no measurable loss of fit quality**:

===========  ==============================
``tol``       alternations per cold start [#]_
===========  ==============================
``1e-6``     ~168
``1e-4``     ~2
===========  ==============================

.. [#] Representative two-component fit on a 24000-pixel spectrum at ``poly_degree = 40``.

Tightening ``tol`` therefore buys sub-noise-level precision at a steep cost in runtime, while loosening
it much further would risk leaving the objective function inconsistent enough to disturb the outer
optimization. ``1e-4`` is the compromise: well below the noise, fast enough to be practical.

Note that a better *a-priori* (initial) polynomial does **not** help here. The convergence is
rate-limited rather than start-limited: because each iteration only shrinks the error by a fixed
fraction, starting closer to the solution still requires essentially the same number of iterations to
reach a given tolerance. The tolerance — not the initialization — is the effective lever.

Non-negative weights
--------------------

Component weights represent physical flux fractions, so they are constrained to be non-negative
(``w_k ≥ 0``) using a bounded least-squares solver — the analogue of ULySS's BVLS. The price of this
constraint is that a combination that would genuinely require a negative weight (e.g. two
near-collinear templates where one must be subtracted) is instead fit with the offending weight
clipped to zero.
