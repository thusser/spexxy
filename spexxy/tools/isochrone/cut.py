import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_parser(subparsers):
    # init parser
    parser = subparsers.add_parser('cut', help='Cut isochrone')
    parser.set_defaults(func=run)

    # parameters
    parser.add_argument('input', help='Input file containing the isochrone', type=str)
    parser.add_argument('output', help='Output file', type=str)
    parser.add_argument('regions', help='Regions to include', type=str, nargs='+',
                        choices=['MS', 'SGB', 'RGB', 'HB', 'AGB'])
    parser.add_argument('-p', '--plot', help='Plot result', action='store_true')


def run(args):
    # load data
    isochrone = pd.read_csv(args.input, index_col=False, comment='#')

    # sort isochrone by M_ini
    isochrone.sort_values('M_ini', inplace=True)

    # plot isochrone?
    if args.plot:
        plt.plot(isochrone['Teff'], isochrone['logg'], label='Input isochrone')

    # find MS by finding first turn-around for Teff
    Teff = isochrone['Teff']
    MS = None
    for i in range(len(Teff) - 1):
        if Teff.iloc[i+1] < Teff.iloc[i]:
            logging.info('Found main sequence turnoff at %.2fK.', Teff.iloc[i])
            MS = isochrone.iloc[:i+1]
            isochrone = isochrone.iloc[i+1:]
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
    isochrone = isochrone.iloc[:i]

    # plot parts
    if args.plot:
        # plots
        plt.scatter(MS['Teff'], MS['logg'], label='MS', c='blue')
        plt.scatter(GB['Teff'], GB['logg'], label='SGB/RGB', c='red')
        plt.scatter(HB['Teff'], HB['logg'], label='HB', c='black')
        #plt.scatter(AGB['Teff'], AGB['logg'], label='AGB')
        # stuff
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend(loc=7)
        plt.xlabel('Teff [K]')
        plt.ylabel('logg [cm/s/s]')
        plt.show()

    # connect parts again
    parts = []
    if 'MS' in args.regions:
        parts += [MS]
    if 'SGB' in args.regions or 'RGB' in args.regions:
        parts += [GB]
    if 'HB' in args.regions:
        parts += [HB]
    #if 'AGB' in args.regions:
    #    parts += [AGB]
    output = pd.concat(parts)

    # write output
    output.to_csv(args.output, index=False)
