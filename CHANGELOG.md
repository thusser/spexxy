## Changelog

### v2.1 (2018-11-26)
* Showing default values in help on command line.
* Moving fixing of parameters in ParamsFit from the constructor into the actual call, so that it is done after
  initializing the components.
* In ParamsFit, always allow parameters that are requested to be fitted to vary in the optimization. This gets important
  for the new "MultiMain", where a once fixed parameter can be allowed to vary in the next fit.
* Introduced new Weight classes, which handle weights for each pixel in a spectrum. Multiple weights are multiplied.
* Added two Weight classes: WeightFromSigma loads the SIGMA array from the FITS file (which acts the same as the
  default behaviour in the previous *spexxy* version), and WeightRanges assigns weights for given wavelength ranges.
* Updated documentation.

### v2.0 (2018-11-16)
* First published *spexxy* version.
