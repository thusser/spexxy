Grids
=====

A "Grid" in *spexxy* is anything that provides any kind of data in a regularly spaced parameter space. The base class
for all Grids is :class:`Grid`, which also defines some convenience methods.

The usual way of getting data from a grid is by calling it with the requested parameters::

    grid = Grid()
    data = grid((3.14, 42.))

A class inheriting from :class:`Grid <spexxy.grid.Grid>` must call Grid's constructor with a list of
:class:`GridAxis <spexxy.grid.GridAxis>` objects that describe the axes of the grid, i.e. their names and possbile
values. Furthermore it must overwrite all necessary methods, in particular
:meth:`__call__() <spexxy.grid.Grid.__call__>`, :meth:`__contains__() <spexxy.grid.Grid.__contains__>`, and
:meth:`all() <spexxy.grid.Grid.all>`. See the implementation of :class:`ValuesGrid <spexxy.grid.ValuesGrid>` for a
simple example.

*spexxy* comes with two pre-defined grids:

* :class:`ValuesGrid <spexxy.grid.ValuesGrid>` defines a simple grid, for which the values are defined in its
  constructor. This grid is mainly used for unit tests.
* :class:`FilesGrid <spexxy.grid.FilesGrid>` is a grid, where each "value" is a spectrum in its own file. A CSV
  file must be provided containing filenames and all parameters.

Grid
----
.. autoclass:: spexxy.grid.Grid
    :members:

GridAxis
--------
.. autoclass:: spexxy.grid.GridAxis
    :members:

ValuesGrid
----------
.. autoclass:: spexxy.grid.ValuesGrid
    :members:

FilesGrid
---------
.. autoclass:: spexxy.grid.FilesGrid
    :members:
