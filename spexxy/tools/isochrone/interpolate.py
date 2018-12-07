import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('interpolate', help='Interpolate isochrone')
    parser.set_defaults(func=run)

    # parameters
    parser.add_argument('input', help='Input file containing the isochrone', type=str)
    parser.add_argument('output', help='Output file', type=str)
    parser.add_argument('--count', help='Number of points for new isochrone', type=int, default=500)
    parser.add_argument('--column', help='Column used for interpolation', type=str, default='M_ini')
    parser.add_argument('--space', help='Space for distance calculations', type=str, nargs=2, default=['Teff', 'logg'])
    parser.add_argument('-p', '--plot', help='Plot result', action='store_true')


def _norm_space(s):
    s -= np.min(s)
    return s / np.max(s)


def run(args):
    # load data
    isochrone = pd.read_csv(args.input, index_col=False, comment='#')

    # get X column for interpolator
    x = isochrone[args.column]
    x_min = np.min(x)
    x_max = np.max(x)

    # normalize space to 0..1
    space_x = _norm_space(isochrone[args.space[0]].values.copy())
    space_y = _norm_space(isochrone[args.space[1]].values.copy())

    # integrate space along isochrone
    logging.info('Creating new grid with %d points in %s between %.3f and %.3f', args.count, args.column, x_min, x_max)
    dist = [0.]
    for i in range(len(space_x) - 1):
        dist += [dist[-1] + np.sqrt((space_x[i+1]-space_x[i])**2. + (space_y[i+1]-space_y[i])**2.)]

    # create interpolator of x on dist
    ip = interp1d(dist, x)

    # interpolate x with constant sampling, which will be our new grid
    grid = ip(np.linspace(0., dist[-1], args.count))

    # get new grid
    data = {args.column: grid}

    # interpolate
    for col in isochrone.columns:
        # skip column from command line
        if col == args.column:
            continue
        logging.info('Interpolating column "%s"...', col)

        # create interpolator
        ip = interp1d(x, isochrone[col].values)

        # do interpolation
        data[col] = ip(grid)

    # create new dataframe
    output = pd.DataFrame(data)

    # write output
    output[isochrone.columns].to_csv(args.output, index=False)

    # plot
    if args.plot:
        plt.scatter(isochrone['Teff'], isochrone['logg'], s=10.0, label='Input isochrone')
        plt.scatter(output['Teff'], output['logg'], s=0.5, label='Output isochrone')
        plt.legend()
        plt.grid()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlabel(args.space[0])
        plt.ylabel(args.space[1])
        plt.show()
