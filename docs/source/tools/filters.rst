filters
=======

The methods in `spexxytools filters` can be used to apply filters.


limbdark
--------

With this command one can apply a given filter to either a single specific intensity spectra or to a grid of those::

    usage: spexxytools filters limbdark [-h] [-g] input output filter

    positional arguments:
      input       Spectrum or grid file (in --grid mode)
      output      Output filename or directory (in --grid mode)
      filter      Filter name or filename

    optional arguments:
      -h, --help  show this help message and exit
      -g, --grid  Process all files in grid file given as input


If `--grid` is given, input must be a grid file and output a directory, otherwise it's input and output filename for
a single spectrum.