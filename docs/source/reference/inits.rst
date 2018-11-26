Inits
=====
Classes derived from the :class:`Init <spexxy.init.Init>` class initialize the parameters of components. After
creating an Init, it is applied to a component by calling it with the component and a filename as parameters.
Therefore a derived class must implement both the constructor :meth:`__init__() <spexxy.init.Init.__init__>` and
:meth:`__call__() <spexxy.init.Init.__call__>`.

*spexxy* comes with a few pre-defined init classes:

* :class:`InitFromCsv <spexxy.init.InitFromCsv>` reads initial values from a CSV file.
* :class:`InitFromPath <spexxy.init.InitFromPath>` looks for a file of the same name in a given path and reads the
  initial values from its FITS header.
* :class:`InitFromValues <spexxy.init.InitFromValues>` takes initial values directly from its constructor.
* :class:`InitFromVhelio <spexxy.init.InitFromVhelio>` takes coordinates and time from the FITS header, and
  calculates the heliocentric or baryiocentric correction, which then can be set as initial value for a
  component.


Init
----
.. autoclass:: spexxy.init.Init
    :members:

InitFromCsv
-----------
.. autoclass:: spexxy.init.InitFromCsv
    :members:

InitFromPath
------------
.. autoclass:: spexxy.init.InitFromPath
    :members:

InitFromValues
--------------
.. autoclass:: spexxy.init.InitFromValues
    :members:

InitFromVhelio
--------------
.. autoclass:: spexxy.init.InitFromVhelio
    :members:
