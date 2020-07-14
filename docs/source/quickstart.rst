Quickstart
==========

Installing *spexxy*
-------------------

The easiest way for installing *spexxy* is using pip::

    pip3 install spexxy

Add the `--user` switch to install it in your user directory (no root required).


Running *spexxy*
----------------

A basic call of *spexxy* is of the form::

    spexxy config.yaml *.fits

which uses the configuration given in `config.yaml` in order to process the files
specified by `*.fits`.

By default, results are written into a CSV file named `spexxy.csv`, but this can be
changed using the `--output` parameter. The CSV file will contain one line per file, each
giving the filename and all results for that file.

A previous fit can be continued by adding the `--resume` switch, which then ignores all files,
for which an entry in the output CSV file exists.

If `--mp <N>` is provided, the processing of the files will be parallelized in N processes,
of which each runs a part of the given files sequentially.


Basic configuration
-------------------

A basic configuration file for *spexxy* might look like this::

    main:
      class: path.to.module.SomeClass
      variable: 123

With this config, *spexxy* tries to instantiante `SomeClass` from package `path.to.module` (which
must therefore be in the PYTHONPATH), and forwards all other parameters next to the `class` element
to its constructor. Then the object is called, i.e. its `__call__()` method is called.


Basic fit example
-----------------

A little more complex example that would actually run a parameter fit on the given spectra looks
like this::

   main:
     class: spexxy.main.ParamsFit
     components:
       star:
         class: spexxy.component.GridComponent
         interpolator:
           phx:
             class: spexxy.interpolator.LinearInterpolator
             grid:
               class: spexxy.grid.FilesGrid
               filename: /path/to/grid.csv
         init:
         - class: spexxy.init.InitFromValues
           values:
             logg: 4.5
             Alpha: 0.
             v: 0
             sig: 0
     fixparams:
       star: [sig, Alpha, logg]

As the main routine, in this case the ParamsFit class is provided with one parameter for its
constructor, which is a list of "components" for the fit.

One component is provided and will be created from the `spexxy.component.Grid` class, which
encapsules a `Grid` or `Interpolator` object, which, in turn, is also given here in the form
of a UlyssInterpolator with its one required parameter. All the objects defined by `class`
attributes will automatically be created by *spexxy*.


References in the configuration
-------------------------------

Note that for better readability, the config file can also be written in the following form::

  grids:
    phxgrid:
      class: spexxy.grid.FilesGrid
      filename: /path/to/grid.csv

  interpolators:
    phx:
      class: spexxy.interpolator.LinearInterpolator
      grid: phxgrid

  components:
    star:
      class: spexxy.component.Grid
      interpolator: phx
      init:
      - class: spexxy.init.InitFromValues
        values:
          logg: 4.5
          Alpha: 0.
          v: 0
          sig: 0

  main:
    class: spexxy.main.ParamsFit
    components: [star]
    fixparams:
      star: [sig, Alpha, logg]

This works, because instead of defining all parameter objects directly in the configuration of a
given class, *spexxy* also supports referencing. The GridComponent requires for its `interpolator`
parameter either an object of type `Interpolator`, or a definition in form of a dictionary
containing a `class` element, or the name of an object that is defined with the `interpolators`
of the configuration. Same works for ParamsFit, which accepts a reference to the component named
`star`, which is defined in the `components` block.
