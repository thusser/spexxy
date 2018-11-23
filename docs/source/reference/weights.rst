Weights
=======
Classes inheriting form :class:`Weight <spexxy.weight.Weight>` create weights arrays for spectra. They work very
similar to :doc:`Masks <masks>`, but instead of returning a boolean mask they return a float array, containing a weight
for every pixel in a spectrum.

*spexxy* comes with a few pre-defined weight classes:

* :class:`WeightFromSigma <spexxy.weight.WeightFromSigma>` creates weights from the spectrum's SIGMA array.
* :class:`WeightRanges <spexxy.weight.WeightRanges>` creates weigths from the given ranges.

Weight
------
.. autoclass:: spexxy.weight.Weight
    :members:

WeightFromSigma
---------------
.. autoclass:: spexxy.weight.WeightFromSigma
    :members:

WeightRanges
------------
.. autoclass:: spexxy.weight.WeightRanges
    :members:
