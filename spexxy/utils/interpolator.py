import logging

from ..interpolator import Interpolator


def create_interpolator(definition: str, logger: logging.Logger = None) -> Interpolator:
    """Create an interpolator from a well defined <definition>, which is
    defined as:
        <definition>:<option>(,<option>)*

    Args:
        definition: Definition of grid.
        logger: If provided, log some status output.

    Returns:
        Instance of Interpolator object.
    """
    from .. import interpolator
    from .. import grid

    # first we split by ':'
    s = definition.split(':')
    if len(s) != 2:
        raise ValueError('Grid definition must be of form <type>:<options>.')

    # check type
    if s[0].lower() == 'ulyss':
        # create ulyss interpolator
        if logger:
            logger.info("Loading ulyss interpolator {0:s}.".format(s[1]))
        return interpolator.UlyssInterpolator(s[1])

    elif s[0].lower() == 'phoenix':
        # further split is by ','
        t = s[1].split(',')

        # load grid
        if logger:
            logger.info('Loading phoenix grid {0:s}.'.format(t[0]))
        phxgrid = grid.PhoenixGrid(t[0])

        # second option would be for the spline interpolator
        if len(t) == 2:
            # if it is 'none', we do not load pre-calculated 2nd derivatives
            if t[1].lower() != 'none':
                if logger:
                    logger.info('Loading phoenix derivatives grid {0:s}.'.format(t[1]))
                derivs = grid.PhoenixGrid(t[1])
            else:
                if logger:
                    logger.warning('Using no pre-calculated derivatives.')
                derivs = None

            # return spline interpolator
            if logger:
                logger.info('Creating spline interpolator.')
            return interpolator.SplineInterpolator(phxgrid, derivs)

        else:
            # just return the linear interpolator
            if logger:
                logger.info('Creating linear interpolator.')
            return interpolator.LinearInterpolator(phxgrid)

    else:
        # invalid grid type
        if logger:
            logger.error("Invalid grid type {0:s}.".format(s[0]))
        raise ValueError("Invalid grid type {0:s}.".format(s[0]))
