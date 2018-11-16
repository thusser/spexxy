from astropy.io import fits
from typing import Union, Tuple, Any


class ResultsFITS(object):
    """Handles results in the primary header of a FITS file."""

    def __init__(self, hdu: fits.PrimaryHDU, namespace: str, prefix: str = 'ANALYSIS'):
        """Initialize a new results set or loads it from a HDU

        Args:
            hdu: HDU of FITS file to store/retrieve results into/from.
            namespace: Namespace for results.
        """
        self._hdu = hdu
        self._prefix = [prefix.upper()]
        self._namespace = self._prefix + namespace.upper().split()
        self._data = {}

        # load from file
        self._load()

    def __getitem__(self, key: str) -> Tuple[float, float]:
        """Returns a single result as (value,error) pair.

        Args:
            key: Name of result.

        Returns:
            Result as (value,error) pair
        """

        # if key doesn't exist, return None
        if key.upper() not in self._data:
            return None

        # get value from dict
        return self._data[key.upper()]

    def __setitem__(self, key: str, value: Union[Tuple, float, str]):
        """Set a new result.

        Args:
            key: Name of result.
            value: Either a value or a list-like value/error pair.
        """

        # just a value or value/error list?
        if hasattr(value, '__iter__') and not isinstance(value, str):
            # if it is a list, we need two entries for value and error
            if len(value) == 2:
                self._data[key.upper()] = value
            else:
                raise ValueError("Value and error must be given.")
        else:
            # just set value with no error
            self._data[key.upper()] = [value, None]

        # save results back to HDU
        self._save()

    def __delitem__(self, key: str):
        """Delete result from set.

        Args:
            key: Name of result.
        """

        # does key exist?
        if key in self._data:
            # delete entry
            del self._data[key]

            # save changes back to HDU
            self._save()

    def __len__(self):
        """Number of results in set."""
        return len(self._data)

    def __iter__(self):
        """Return iterator over all result names."""
        return iter(self._data.keys())

    def __contains__(self, item):
        """Whether results set contains the given item."""
        return item.upper() in self._data

    def keys(self):
        """Returns list of keys in result set."""
        return self._data.keys()

    @property
    def data(self):
        """Returns results set as dict."""
        return self._data

    def _load(self):
        """Load results set from HDU."""

        # init
        self._data = {}
        s = len(self._namespace)

        # get header
        hdr = self._hdu.header.cards

        # loop all entries
        for h in hdr:
            # split keyword
            key = str(h.keyword).split()

            # is this the namespace we want?
            if key[:s] == self._namespace:
                # get name
                name = " ".join(key[s:-1] if key[-1] == "ERR" else key[s:])

                # check data entry
                if name not in self._data:
                    self._data[name] = [None, None]

                # set it
                if key[-1] == "ERR":
                    self._data[name][1] = h.value
                else:
                    self._data[name][0] = h.value

    def _update_header(self, key: str, value: Any):
        """Update HDU header keyword.

        Args:
            key: Name of header keyword.
            value: New value for header entry.
        """

        # does keyword exist?
        if key.upper() in list(self._hdu.header.keys()):
            self._hdu.header[key.upper()] = value
        else:
            self._hdu.header["HIERARCH " + key.upper()] = value

    def _save(self):
        """Write result set back to HDU header."""

        # number of text messages written
        num_msg = 1

        # first remove all keys of this namespace
        hdr = self._hdu.header
        for key in list(hdr.keys()):
            # base string for this key
            base = "%s" % " ".join(self._namespace)
            if str(key.strip()).startswith(base):
                del hdr["HIERARCH " + key.strip()]

        # loop all data
        for key in self._data.keys():
            # base string for this key
            base = str("%s %s" % (" ".join(self._namespace), key))

            # write value
            if type(self._data[key][0]) is str:
                # create new MSG entry, if value is a string
                self._update_header(base, "MSG%04d" % num_msg)
                hdr["MSG%04d" % num_msg] = self._data[key][0]
                num_msg += 1
            else:
                # just write the value
                self._update_header(base, self._data[key][0])

            # write error
            if self._data[key][1] is not None:
                self._update_header(base + " ERR", self._data[key][1])


__all__ = ['ResultsFITS']
