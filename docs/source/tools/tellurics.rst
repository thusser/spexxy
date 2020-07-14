tellurics
=========

The methods in `spexxytools tellurics` are used for handling tellurics.


mean
----

Calculates a mean tellurics component from TELLURICS extensions in a list of files, e.g. from a ParamsFit run::

    usage: spexxytools tellurics mean [-h] [-o OUTPUT] [-l SNLIMIT] [-f SNFRAC] [-w] [-r RESAMPLE RESAMPLE RESAMPLE] spectra [spectra ...]

    positional arguments:
      spectra               List of processed spectra

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            File to write tellurics to
      -l SNLIMIT, --snlimit SNLIMIT
                            minimum SNR to use
      -f SNFRAC, --snfrac SNFRAC
                            if snlimit is not given, calculate it from X percent best
      -w, --weight          weight by SNR
      -r RESAMPLE RESAMPLE RESAMPLE, --resample RESAMPLE RESAMPLE RESAMPLE
                            resample to start,step,count
