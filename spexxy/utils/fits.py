import glob
import io
import os
import subprocess
from typing import Union, List
import pandas as pd
import numpy as np


def bulk_read_header(folder_or_files: Union[str, List], keys: Union[str, List[str]], pattern: str = '*.fits',
                     chunksize: int = 10000) -> pd.DataFrame:
    """
    A wrapper around the dfits/fitsort command-line tools to query a list of
    FITS-files for one or more header keywords and store their values in a
    pandas DataFrame.

    Args:
        folder_or_files: List of files or the folder to work in.
        keys: The header keyword(s) that should be queried.
        pattern: A common pattern by which the input FITS-files can be recognized.
        chunksize: Maximum number of files to process per system call.

    Returns:
        The values of the requested keywords in each FITS-file. Note that
        currently no conversion is performed on the columns, this has to be
        done separately. For example, to convert the values of column 'XXX'
        to numeric values, one can do:
            output['XXX'] = output['XXX'].apply(pd.to_numeric, errors='coerce')
    """

    # get list of files
    if isinstance(folder_or_files, str):
        files = [os.path.split(f)[1] for f in glob.glob(os.path.join(folder, pattern))]
    elif isinstance(folder_or_files, list):
        files = folder_or_files
    else:
        raise ValueError('Unknown format.')

    # no chunksize given?
    if chunksize is None:
        chunksize = len(files)

    # prepare results
    results = []

    # loop chunks
    for start in np.arange(0, len(files), chunksize):
        # get subset
        sub = files[start:start + chunksize]

        # prepare dfits call
        arguments = ['dfits']
        arguments.extend(sub)
        dfits = subprocess.Popen(arguments, stdout=subprocess.PIPE, cwd=folder)

        # prepare fitsort call
        arguments = ['fitsort']
        if hasattr(keys, "__iter__") and not isinstance(keys, str):
            for key in keys:
                arguments.append(key)
        else:
            arguments.append(keys)
        fitsort = subprocess.Popen(arguments, stdin=dfits.stdout, stdout=subprocess.PIPE)

        # get output
        result = fitsort.communicate()[0]

        # convert output to pandas dataframe
        output = pd.read_table(io.StringIO(result.decode('utf-8')))
        for column in output.columns:
            if column[:7] == 'Unnamed':  # ignore columns named 'Unnamed ...'
                del output[column]
            elif column != column.strip():  # remove leading/trailing whitespaces in column names
                output.rename(columns={column: column.strip()}, inplace=True)

        # add to list
        results.append(output)

    # return concatenated results
    return pd.concat(results)
