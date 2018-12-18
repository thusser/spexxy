import logging
import math
import re
import io

import numpy as np
import pandas as pd
from scipy import optimize


class Isochrone:
    """Handles an isochrone in spexxy."""
    def __init__(self, data: pd.DataFrame, meta: dict = None):
        """Create a new isochrone.
        
        Args:
            data: Actual isochrone data with column Teff, logg, logL/Lo, mbol, M_ini, M_act, and one for each filter.
            meta: Meta data for the isochrone, i.e. age, metallicity, etc.
        """
        self._data = data
        self._meta = meta

    @property
    def filters(self) -> list:
        """Returns a list of filters defines by this isochrone.

        Returns:
            List of filters.
        """
        return list(self._data.columns.values[6:])

    @property
    def data(self) -> pd.DataFrame:
        """Returns the full isochrone data.

        Returns:
            Isochrone data table.
        """
        return self._data

    def __getitem__(self, item: str) -> float:
        """Returns a meta data item.

        Args:
            item: Name of meta data.

        Returns:
            Meta data value.
        """
        return self._meta[item] if item in self._meta else None

    def __setitem__(self, item: str, value: float):
        """Sets a given meta data item.

        Args:
            item: Name of meta data.
            value: New value for meta data.
        """
        self._meta[item] = value

    def copy(self) -> 'Isochrone':
        """Create a deep copy of this isochrone.

        Returns:
            Copy of this isochrone.
        """
        return Isochrone(self._data.copy(), dict(self._meta))

    @staticmethod
    def load(filename: str) -> 'Isochrone':
        """Load an isochrone from file.

        Args:
            filename: Filename of isochrone.

        Returns:
            Loaded isochrone.
        """

        # try to read meta data
        meta = {}
        with open(filename, 'r') as f:
            for line in f:
                # we break on first non-comment line
                if not line.startswith('#'):
                    break

                # split by first '=', if exists
                eq = line.find('=')
                if eq > -1:
                    # get it
                    key = line[1:eq].strip()
                    val = line[eq+1:].strip()

                    # is val a float?
                    try:
                        val = float(val)
                    except ValueError:
                        pass

                    # store it
                    meta[key] = val

        # load data
        data = pd.read_csv(filename, index_col=False, comment='#')

        # return new Isochrone
        return Isochrone(data, meta)

    def save(self, filename: str):
        """Write isochrone to a file.

        Args:
            filename: Name of file to write isochrone in.
        """

        # open file
        with open(filename, 'w') as f:
            # write all meta data
            for key, val in self._meta.items():
                f.write('# %s=%s\n' % (key, str(val)))

            # write data (for some reason this StringIO workaround seems to be necessary sometimes...
            with io.StringIO() as sio:
                self._data.to_csv(sio, index=False)
                f.write(sio.getvalue())

    @staticmethod
    def _polynomial(x, y, p):
        """Polynomial that can be fitted to an isochrone.

        Args:
            x: Colours of points in isochrone.
            y: Magnitudes of points in isochrone.
            p: Coefficients for polynomial.

        Returns:
            Evaluated polynomial.
        """
        return p[0] + p[1] * x + p[2] * y + p[3] * x * x + p[4] * y * y + p[5] * x * y + p[6] * x * x * x + \
               p[7] * x * x * y + p[8] * x * y * y + p[9] * y * y * y

    def _fit(self, col: list, mag: list, value: list):
        """Fit a polynomial to the isochrone.

        Args:
            col: List (or equivalent) of colours to fit.
            mag: List (or equivalent) of magnitudes to fit.
            value: List (or equivalent) of data to fit.

        Returns:

        """
        # error function
        func = lambda p: self._polynomial(col, mag, p) - value

        # initial guess
        initial = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # fit
        res = optimize.leastsq(func, initial, full_output=True)
        if res[4] not in [1, 2, 3, 4]:
            raise ValueError("Could not fit isochrone: " + res[3])

        # return parameters
        return res[0]

    def interpolator(self, column: str, filter1: str, filter2: str):
        """Create a polynomial interpolator for the given data column and two filters.

        Args:
            column: Name of column containing data.
            filter1: Name of first filter, used for magnitude.
            filter2: Name of second filter, colour is calculated as filter1-filter2.

        Returns:
            Function for interpolating isochrone with a polynomial.
        """

        # create colour column and get mag
        colour = self._data[filter1] - self._data[filter2]
        magnitudes = self._data[filter1]

        # fit polynomial
        poly = self._fit(colour, magnitudes, self._data[column])

        # define method to return
        def interpolator_inner(col, mag):
            return self._polynomial(col, mag, poly)

        # return method
        return interpolator_inner

    def nearest(self, filter1: str, filter2: str):
        """Creates an 'interpolator' that returns the nearest point on the isochrone for a colour/magnitude pair.

        Args:
            filter1: Name of first filter, used for magnitude.
            filter2: Name of second filter, colour is calculated as filter1-filter2.

        Returns:
            Function that calculates nearest neighbours on isochrone.
        """

        # create colour column and get mag
        colour = self._data[filter1] - self._data[filter2]
        magnitudes = self._data[filter1]

        # define method to return
        def nearest_inner(col, mag):
            # check
            if col is None or mag is None:
                return None

            # get nearest, scale colour with factor 7
            tmp = 49. * (colour - col)**2. + (magnitudes - mag)**2.
            d = np.argmin(tmp)

            # return values
            return {
                'dist': math.sqrt(tmp[d]),
                'v-i': self._data.loc[d, filter1] - self._data.loc[d, filter2],
                'v': self._data.loc[d, filter1],
                'teff': self._data.loc[d, 'Teff'],
                'logg': self._data.loc[d, 'logg'],
                'mini': self._data.loc[d, 'M_ini'],
                'mact': self._data.loc[d, 'M_act']
            }

        # return inner method
        return nearest_inner

    def apply_distance(self, distance: float):
        """Apply a distance in pc to the isochrone.

        Args:
            distance: Distance in pc.
        """

        # calculating distance modulus
        dist_mod = 5. * np.log10(distance) - 5.

        # add to filter columns
        self._data[self.filters] += dist_mod

    def keep_filters(self, filters: list):
        """Drops all filters except the given ones.
        
        Args:
            filters: List of filters to keep.
        """

        # loop all filters
        for f in list(self.filters):
            if f not in filters:
                self._data.drop(f, axis=1, inplace=True)

    @staticmethod
    def import_cmd27(filename) -> 'Isochrone':
        """Import PARSEC isochrone in version 2.7.

        Args:
            filename: Name of file to load.

        Returns:
            Parsed isochrone.
        """

        # regular expression for search for end of header
        re_hdr = re.compile(r'\[M/H\]\s+=\s+([+-]?[0-9]+\.[0-9]+).*Age\s+=\s+([0-9]*\.[0-9]*e[+-][0-9]*)\s+yr')

        # find line with header
        header_lines = None
        m_h = None
        age = None
        header = None
        logging.info('Searching for header...')
        with open(filename, "r") as f:
            for header_lines, line in enumerate(f):
                m = re_hdr.search(line)
                if m:
                    m_h = float(m.group(1))
                    age = float(m.group(2))

                if line[0] == '#' and 'log(age/yr)' in line:
                    # found last header line, read header
                    header = line[1:].split()
                    break

        # no header or [M/H] found?
        if header is None:
            raise IOError('Could not find header.')
        if m_h is None or age is None:
            raise IOError('Could not find [M/H] or age.')
        logging.info('Found Age=%.2fgyr, [M/H]=%.2f.', age/1e9, m_h)

        # read file
        logging.info('Reading data...')
        data = pd.read_csv(filename, skiprows=header_lines + 1, sep='\s+', names=header)

        # find filter columns
        filters = list(data.columns.values[8:-2])
        logging.info('Found filters: ' + ', '.join(filters))

        # we don't want log(Teff), but Teff
        data['Teff'] = 10. ** data['logTe']

        # get meta data
        meta = {
            'Age': age,
            '[M/H]': m_h
        }

        # rename some columns
        data = data.rename(columns={'logG': 'logg'})

        # log it
        logging.info('Found columns: ' + ', '.join(data.columns))

        # return isochrone
        return Isochrone(data[['Teff', 'logg', 'logL/Lo', 'mbol', 'M_ini', 'M_act'] + filters], meta)

    @staticmethod
    def import_cmd29(filename: str) -> Isochrone:
        """Import PARSEC isochrone in version 2.9.

        Args:
            filename: Name of file to load.

        Returns:
            Parsed isochrone.
        """

        # find line with header
        header = None
        logging.info('Searching for header...')
        with open(filename, "r") as f:
            last_line = None
            for line in f:
                # first line with no comment? means last line was header.
                if not line.startswith('#'):
                    header = last_line[1:].split()
                    break

                # store last line
                last_line = line

        # no header found?
        if header is None:
            raise IOError('Could not find header.')

        # read file
        logging.info('Reading data...')
        data = pd.read_csv(filename, comment='#', sep='\s+', names=header)

        # add metallicity column
        data['FeH'] = np.log(data['Zini'] / 0.0152)

        # remove "*mag" from columns
        data.columns = [c[:-3] if c.endswith('mag') else c for c in data.columns]

        # find filter columns
        filters = list(data.columns.values[24:])
        logging.info('Found filters: ' + ', '.join(filters))

        # we don't want log(Teff), but Teff
        data['Teff'] = 10. ** data['logTe']

        # rename some columns
        data = data.rename(columns={'Mass': 'M_act'})
        data = data.rename(columns={'Mini': 'M_ini'})
        data = data.rename(columns={'logL': 'logL/Lo'})

        # get unique ages and metallicities
        uage = sorted(data['Age'].unique())
        ufeh = sorted(data['FeH'].unique())

        # loop
        isochrones = []
        for age in uage:
            for feh in ufeh:
                # get subset
                d = data[(data['Age'] == age) & (data['FeH'] == feh)]
                if len(d) == 0:
                    continue

                # get age and metallicity
                logging.info('Found Age=%.2fgyr, [M/H]=%.2f.', age/1e9, feh)

                # get meta data
                meta = {
                    'Age': age,
                    '[M/H]': feh
                }

                # create isochrone
                iso = Isochrone(d[['Teff', 'logg', 'logL/Lo', 'mbol', 'M_ini', 'M_act'] + filters], meta)
                isochrones.append(iso)

        # finished
        return isochrones[0] if len(isochrones) == 1 else isochrones

    def cut_regions(self, regions: list):
        """Tries to find regions in isochrone and cuts it down to the defined regions.

        Args:
            regions: List of region names (MS, SGB, RGB, HB).
        """

        # sort isochrone by M_ini
        isochrone = self._data[:]
        isochrone.sort_values('M_ini', inplace=True)

        # find MS by finding first turn-around for Teff
        Teff = isochrone['Teff']
        MS = None
        for i in range(len(Teff) - 1):
            if Teff.iloc[i + 1] < Teff.iloc[i]:
                logging.info('Found main sequence turnoff at %.2fK.', Teff.iloc[i])
                MS = isochrone.iloc[:i + 1]
                isochrone = isochrone.iloc[i + 1:]
                break

        # find tip of RGB by finding first turn-off for Teff at logg<1
        GB = None
        logg = isochrone['logg']
        Teff = isochrone['Teff']
        for i in range(len(Teff) - 1):
            if logg.iloc[i] < 1 and Teff.iloc[i + 1] > Teff.iloc[i]:
                logging.info('Found tip of RPG at %.2fK.', Teff.iloc[i])
                GB = isochrone.iloc[:i + 1]
                isochrone = isochrone.iloc[i + 1:]
                break

        # find HB by finding highest temperature
        Teff = isochrone['Teff']
        i = np.argmax(Teff.values)
        logging.info('Found end of horizontal branch at %.2fK.', Teff.iloc[i])
        HB = isochrone.iloc[i:]

        # connect parts again
        parts = []
        if 'MS' in regions:
            parts += [MS]
        if 'SGB' in regions or 'RGB' in regions:
            parts += [GB]
        if 'HB' in regions:
            parts += [HB]
        # if 'AGB' in regions:
        #    parts += [AGB]
        self._data = pd.concat(parts)


__all__ = ['Isochrone']
