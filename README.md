## *spexxy*

```text
___ _ __   _____  ____  ___   _ 
/ __| '_ \ / _ \ \/ /\ \/ / | | |
\__ \ |_) |  __/>  <  >  <| |_| |
|___/ .__/ \___/_/\_\/_/\_\\__, |
    | |     A spectrum      __/ |
    |_| fitting framework  |___/ 
```


![develop build](https://github.com/thusser/spexxy/workflows/pytest/badge.svg)

*spexxy* is a framework for the analysis of astronomical spectra. It provides both two executables
that wrap the framework using a YAML configuration file and provide some additional command line
tools, respectively.

From the beginning, *spexxy* has been strongly influenced by [ULySS](http://ulyss.univ-lyon1.fr/). The present
version 2 of *spexxy* is not a complete rewrite of the original version, but a highly refactored and optimized 
re-release.


### Documentation

You can find the documentation for spexxy at <https://spexxy.readthedocs.io/>.

### Versions
See [Changelog](CHANGELOG.md).
* v2.3 (2019-11-15)
* v2.2 (2019-02-04)
* v2.1 (2018-11-26)
* v2.0 (2018-11-16)


### References

*spexxy* requires Python 3.6 or later and depends on a couple of amazing external packages:

* [Astropy](http://www.astropy.org/) is used for handling FITS file, times and coordinates, and other
  astronomical calculations.
* [LMFIT](https://lmfit.github.io/lmfit-py/) is a central part of *spexxy*, for it handles the main
  optimization routines.
* [NumPy](http://www.numpy.org/) is mainly used for array handling.
* [pandas](https://pandas.pydata.org/) provides easy access to CSV files and allows for easy table
  handling.
* [PyYAML](https://pyyaml.org/) adds support for YAML configuration files.
* [SciPy](https://www.scipy.org/) is used for optimization, interpolation, and integration at several
  places.

Thanks to everyone putting time and efforts into these (and other!) open source projects!
