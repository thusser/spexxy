import os


def data_filename(filename: str) -> str:
    """Returns the absolute path of a file in the testdata directory.

    Parameters
    ----------
    filename : str
        Relative filename

    Returns
    -------
    str
        Absolute filename
    """
    return os.path.join(os.path.dirname(__file__), filename)
