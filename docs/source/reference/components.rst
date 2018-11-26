Components
==========

A "Component" in *spexxy* describes a component in a fit, which in general is a collection of parameters and a method
to fetch data for a given set of parameter values. Typically, a component build on top of a :doc:`Grid <grids>` or
an :doc:`interpolator <interpolators>`.

*spexxy* comes with a few pre-defined grids:

* :class:`SpectrumComponent <spexxy.component.SpectrumComponent>` is the base class for all components that serve
  spectra.
* :class:`StarComponent <spexxy.component.StarComponent>` contains a single spectrum and adds LOSVD parameters.
* :class:`GridComponent <spexxy.component.GridComponent>` wraps a grid or an interpolator into a component and adds
  LOSVD parameters.
* :class:`TelluricsComponent <spexxy.component.GridComponent>` is just a convenient class derived from
  `GridComponent <spexxy.component.GridComponent>` that changes the default's component name.

Component
---------
.. autoclass:: spexxy.component.Component
    :members:

SpectrumComponent
-----------------
.. autoclass:: spexxy.component.SpectrumComponent
    :members:

StarComponent
-------------
.. autoclass:: spexxy.component.StarComponent
    :members:

GridComponent
-------------
.. autoclass:: spexxy.component.GridComponent
    :members:

TelluricsComponent
------------------
.. autoclass:: spexxy.component.TelluricsComponent
    :members:
