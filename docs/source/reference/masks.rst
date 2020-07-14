Masks
=====
Classes inheriting form :class:`Mask <spexxy.mask.Mask>` create good pixel masks for spectra. They are called
by the main routine with each component as parameter. So in order to work, derived classes must implement both
the constructor :meth:`__init__() <spexxy.mask.Mask.__init__>` and :meth:`__call__() <spexxy.mask.Mask.__call__>`.

*spexxy* comes with a few pre-defined mask classes:

* :class:`MaskEnds <spexxy.mask.MaskEnds>` masks a given number of pixels at both ends of a spectrum.
* :class:`MaskFromPath <spexxy.mask.MaskFromPath>` loads a mask from a file with the same name in a given directory.
* :class:`MaskRanges <spexxy.mask.MaskRanges>` accepts wavelength ranges in its constructor that are then used to
  create a mask.

Mask
----
.. autoclass:: spexxy.mask.Mask
    :members:

MaskEnds
--------
.. autoclass:: spexxy.mask.MaskEnds
    :members:

MaskFromPath
------------
.. autoclass:: spexxy.mask.MaskFromPath
    :members:

MaskNegative
------------
.. autoclass:: spexxy.mask.MaskNegative
    :members:

MaskRanges
----------
.. autoclass:: spexxy.mask.MaskRanges
    :members:
