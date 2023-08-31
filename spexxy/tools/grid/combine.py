import logging
import argparse
import os
import shutil
from typing import List

from spexxy.grid import Grid

log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('combine', help='Combine multiple grids into one',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output', type=str, help='Output path')
    parser.add_argument('sources', type=str, nargs='+', help='Input CSV files')

    # argparse wrapper for combine_grid
    def run(args):
        combine_grid(**vars(args))
    parser.set_defaults(func=run)


def combine_grid(output: str, sources: List[str], **kwargs):
    # load first grid
    grid = Grid.load(sources[0])

    # outdir and grid file
    if not os.path.exists(output):
        os.makedirs(output)
    with open(os.path.join(output, 'grid.csv'), 'w') as f:
        f.write('Filename,' + ','.join(grid.axis_names()) + '\n')

    # store existing parameters
    existing_params = []

    # loop all input grids
    for src in sources:
        # load grid
        grid = Grid.load(src)

        # only filegrids supported
        if not hasattr(grid, 'filename'):
            print('Only filegrids supported.')
            continue

        # get all params
        all_params = grid.all()

        # loop params
        for params in all_params:
            # does it exist?
            if params in existing_params:
                continue

            # get name and directory of output file
            filename = grid.filename(params)
            basepath = os.path.relpath(filename, os.path.dirname(src))
            outfile = os.path.join(output, basepath)
            outpath = os.path.dirname(outfile)

            # does dir exist?
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # copy
            log.info("  Copying to {0:s}...".format(outfile))
            shutil.copyfile(filename, outfile)

            # add to CSV
            with open(os.path.join(output, 'grid.csv'), 'a') as csv:
                csv.write(filename + ',' + ','.join([str(p) for p in params]) + '\n')

            # add params
            existing_params.append(params)
