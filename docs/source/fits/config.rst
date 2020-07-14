Configuration files
===================

General structure
-----------------

A configuration for spexxy must be provided as YAML file, e.g. something like this::

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
      class: spexxy.component.GridComponent
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

From this example, two observations can be made, which will both be discussed in more detail later:

1. The configuration has several top-level entries (grids, interpolators, ...), of which one -- main -- has a special
   meaning.

2. At several places, Python class names are given. Objects from these classes are created automatically during runtime,
   which allows for changing the behaviour of the fits.


Object creation
---------------

In the above example, several Python class names are given, from which objects are automatically created during runtime.
Values on the same level as the class definition are passed to the class' constructor, e.g.:

1. The class :class:`spexxy.grid.FilesGrid` has a constructor with a single parameter, `filename`, which is given in
   the config.

2. The class :class:`spexxy.component.GridComponent` has two parameters (`interpolator`, `name`) in its constructor and
   gets four more from its parent class :class:`spexxy.component.SpectrumComponent` and three more
   from its parent (:class:`spexxy.component.Component`) in turn (note that `name` is just passed through).
   Except for `interpolator` all parameters are optional, so that one must be given in the config. According to the
   definition in :class:`spexxy.component.Component`, `init` must be a list of :class:`spexxy.init.init.Init` objects,
   which is provided.

In the last case, the value provided for `interpolator` is a string, and not a :class:`spexxy.interpolator.Interpolator`
as required. This works, because *spexxy* usually accepts objects in three forms:

1. An object of the required type directly.

2. A dictionary with a `class` element describing the object.

3. A name of an object described somewhere else. It searches for the name in given groups, which are represented by
   top-level entries in the config.


Top-level entries
-----------------

Only one of the top-level entries in the configuration files must always exist: `main`. It describes a sub-class of
:class:`spexxy.main.MainRoutine` and is the entry point into any (fitting) routine in *spexxy*.

All the other top-level entries define groups, which are mainly used by the main routine
:class:`spexxy.main.ParamsFit`. As described in the section above, objects can be referenced by their name, which is
especially interesting, if they are supposed to be used multiple times, e.g. two different components could use
the same interpolator.

Currently five different groups are in use:

- grids

- interpolators

- components

- weights

- masks

They can always be used in place of parameters with the same name. In the example at the top of this page this
would be:

1. The `grid` parameter for :class:`spexxy.interpolator.LinearInterpolator` refers to an object in the `grids` group
   with the name `phxgrid`.

2. The `interpolator` parameter for :class:`spexxy.component.GridComponent` points to the interpolator in the
   `interpolators' group.

3. The `components` parameter for :class:`spexxy.main.ParamsFit` contains a list with a single name of an element in
   the `components` group.