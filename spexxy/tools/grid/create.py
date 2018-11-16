import logging


log = logging.getLogger(__name__)


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('create', help='Create a new DataBase grid')
    parser.add_argument('root', type=str, help='Path to create grid from', default='.')
    parser.add_argument('-o', '--output', type=str, help='Output database file', default='grid.csv')
    parser.add_argument('-p', '--pattern', type=str, help='Filename pattern', default='**/*.fits')
    parser.add_argument('--from-filename', type=str, help='Parse parameters from the filenames with the given '
                                                          'regular expression')
    # argparse wrapper for create_grid
    def run(args):
        create_grid(**vars(args))
    parser.set_defaults(func=run)


def create_grid(root, output, pattern, from_filename, **kwargs):
    import os
    import glob
    import re

    # easiest way to list all files, is to remember current directory, go to root, glob, and go back
    cwd = os.getcwd()
    os.chdir(root)
    files = sorted(glob.glob(pattern, recursive=True))
    os.chdir(cwd)

    # how do we get the parameters?
    if from_filename is not None:
        # create regexp
        regexp = re.compile(from_filename)

        # get parameters
        parameters = list(regexp.groupindex.keys())

        # open file
        with open(output, 'w') as csv:
            csv.write('Filename,%s\n' % ','.join(parameters))

            # loop files
            for filename in files:
                # do matching
                m = regexp.search(filename)
                if m is None:
                    continue

                # get values by looping values of groupindex and get values of corresponding groups
                # this works in the correct order, because Python guarantees, that keys() and values() on a dict
                # have the same order
                # the +0 avoids negative zeros
                values = [0. if m.group(idx) is None else float(m.group(idx)) + 0. for idx in regexp.groupindex.values()]

                # write to csv
                csv.write('%s,%s\n' % (filename, ','.join([str(v) for v in values])))
