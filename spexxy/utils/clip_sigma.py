import numpy as np


def clip_sigma(data: np.ndarray, kappa_low: float = 5, kappa_high: float = 5, maxiter: int = 99) -> np.ndarray:
    """Kappa-sigma clipping on numpy array.

    Args:
        data: Input array.
        kappa_low: Lower kappa.
        kappa_high: Upper kappa.
        maxiter: Maximum number of iterations.

    Returns:
        Clipped array.
    """

    # do iterations
    for i in range(maxiter):
        # calculate current median and sigma
        median = np.nanmedian(data)
        sigma = np.nanstd(data)

        # create mask
        mask = (data < (median - kappa_low * sigma)) | (data > (median + kappa_high * sigma))

        # no new pixels masked?
        if np.sum(mask) == 0:
            break

        # apply mask
        data = data[~mask]

    # return clipped array
    return data
