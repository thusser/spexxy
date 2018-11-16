Interpolators
=============

A "Interpolator" in *spexxy* is similar to a :doc:`Grid <grids>`, but works on a continuous parameter space instead of
a discrete one. In fact, many interpolators build on an existing grid and allows for interpolation between grid points.

As with a :class:`Grid <spexxy.grid.Grid>`, the usual way of getting data from an interpolator is by calling it with
the requested parameters::

    ip = Interpolator()
    data = ip((3.14, 42.))

A class inheriting from :class:`Interpolator <spexxy.interpolator.Interpolator>` must overwrite all necessary methods,
in particular :meth:`__call__() <spexxy.interpolator.Interpolator.__call__>` and
:meth:`axes() <spexxy.interpolator.Interpolator.axes>`.

*spexxy* comes with three pre-defined interpolators:

* :class:`LinearInterpolator <spexxy.interpolator.LinearInterpolator>` performs linear interpolation on a given grid.
* :class:`SplineInterpolator <spexxy.interpolator.SplineInterpolator>` performs a cubic spline interpolation on a given
  grid.
* :class:`UlyssInterpolator <spexxy.interpolator.UlyssInterpolator>` extracts spectra from interpolator files
  created for the spectrum fitting package `ULySS <http://ulyss.univ-lyon1.fr/>`_.

Interpolator
------------
.. autoclass:: spexxy.interpolator.Interpolator
    :members:

LinearInterpolator
------------------
.. autoclass:: spexxy.interpolator.LinearInterpolator
    :members:

SplineInterpolator
------------------
.. autoclass:: spexxy.interpolator.SplineInterpolator
    :members:

UlyssInterpolator
-----------------
.. autoclass:: spexxy.interpolator.UlyssInterpolator
    :members:
