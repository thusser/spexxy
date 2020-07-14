spectrum
========

The methods in `spexxytools spectrum` are used for handling spectra.


extract
-------

With this tool a given wavelength range can be extracted from a list of spectra::

    usage: spexxytools spectrum extract [-h] [-p PREFIX] start end input [input ...]

    positional arguments:
      start                 Start of wavelength range
      end                   End of wavelength range
      input                 Input spectrum

    optional arguments:
      -h, --help            show this help message and exit
      -p PREFIX, --prefix PREFIX
                            Output file prefix (default: extracted_)
