grid
=====

The methods in `spexxytools grid` can be used to create and manipulate :class:`FilesGrid <spexxy.grid.FilesGrid>`
grids.


info
----
Information about an existing grid can be inspected using `spexxytools grid info` with a grid configuration as
paramater::

    usage: spexxytools grid info [-h] [-v] config

    positional arguments:
      config        Config for grid

    optional arguments:
      -h, --help    show this help message and exit
      -v, --values  Show all values on axes (default: False)

The file specified in the parameters must contain the configuration for a grid (like) object, e.g. something like this::

    class: spexxy.grid.FilesGrid
    filename: /path/to/grid.csv


create
------
Using `spexxytools grid create` a new :class:`FilesGrid <spexxy.grid.FilesGrid>` can be created from files in
a directory::

    usage: spexxytools grid create [-h] [-o OUTPUT] [-p PATTERN]
                                   [--from-filename FROM_FILENAME]
                                   root

    positional arguments:
      root                  Path to create grid from

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Output database file (default: grid.csv)
      -p PATTERN, --pattern PATTERN
                            Filename pattern (default: **/*.fits)
      --from-filename FROM_FILENAME
                            Parse parameters from the filenames with the given
                            regular expression (default: None)

For example, the name of a spectrum in the `PHOENIX library <http://phoenix.astro.physik.uni-goettingen.de/>`_
looks something like this::

    lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

The numbers in the filename are Teff, logg and [M/H], respectively. For alpha element abundances [alpha/Fe]!=0,
another term appears like this::

    lte05800-4.50-0.0.Alpha=+0.50.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

Since all parameters are encoded in the filename, a grid can easily be defining a regular expression that extracts
these parameters from the filenames. Using the spexxytools this can be done as::

    spexxytools grid create --from-filename "lte(?P<Teff>\d{5})-(?P<logg>\d\.\d\d)(?P<FeH>[+-]\d\.\d)(\.Alpha=(?P<Alpha>[+-]\d\.\d\d))?\.PHOENIX" .

Note that the groups are named (via ?P<name>) and that those names will be used as names for the parameters.

This spexxytools call will produce an output CSV file that might look like this::

    Filename,Teff,logg,FeH,Alpha
    PHOENIX-ACES-AGSS-COND-2011/Z+0.5.Alpha=+0.50/lte02300-0.00+0.5.Alpha=+0.50.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits,2300.0,0.0,0.5,0.5
    PHOENIX-ACES-AGSS-COND-2011/Z+0.5.Alpha=+0.50/lte02300-0.50+0.5.Alpha=+0.50.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits,2300.0,0.5,0.5,0.5
    [...]


fillholes
---------
The :doc:`Interpolators <interpolators>` in *spexxy* work best on grids with a convex shape without any holes.
With `spexxytools grid create` there is a tool for filling holes in a grid using a cubic spline interpolator::

    usage: spexxytools grid fillholes [-h] grid output

    positional arguments:
      grid        Name of grid CSV
      output      Directory to store interpolated spectra in

    optional arguments:
      -h, --help  show this help message and exit

