isochrone
=========

The methods in `spexxytools isochrone` can be used to manipulate and apply isochrones.


apply
-----

With this command one can apply a given isochrone to all stars in a given photometry file. The tool derives
effective temperatures, surface gravities and current masses for all entries::

    usage: spexxytools isochrone apply [-h] [-o OUTPUT] [--id ID [ID ...]] [--filter1 FILTER1] [--filter2 FILTER2] [--filter1-iso FILTER1_ISO] [--filter2-iso FILTER2_ISO] [--nearest] [--check-teff CHECK_TEFF CHECK_TEFF] [--check-logg CHECK_LOGG CHECK_LOGG] [--check-mact CHECK_MACT CHECK_MACT]
                                       [--quadratic | --cubic]
                                       isochrone photometry

    positional arguments:
      isochrone             File containing the isochrone
      photometry            File containing the photometry

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Output file
      --id ID [ID ...]      IDs from photometry file to copy to the output
      --filter1 FILTER1     First filter to use
      --filter2 FILTER2     Second filter to use
      --filter1-iso FILTER1_ISO
                            First filter in isochrone to use (--filter1 if empty)
      --filter2-iso FILTER2_ISO
                            Second filter in isochrone to use (--filter2 if empty)
      --nearest             Use nearest neighbour instead of polynomial
      --check-teff CHECK_TEFF CHECK_TEFF
                            Only apply values if Teff within range
      --check-logg CHECK_LOGG CHECK_LOGG
                            Only apply values if logg within range
      --check-mact CHECK_MACT CHECK_MACT
                            Only apply values if Mact within range
      --quadratic           Use quadratic polynomial instead of linear one
      --cubic               Use cubic polynomial instead of linear one


cut
---

Tries to automatically cut an isochrone to given regions (MS, RGB, ...)::

    usage: spexxytools isochrone cut [-h] [-p] input output {MS,SGB,RGB,HB,AGB} [{MS,SGB,RGB,HB,AGB} ...]

    positional arguments:
      input                Input file containing the isochrone
      output               Output file
      {MS,SGB,RGB,HB,AGB}  Regions to include

    optional arguments:
      -h, --help           show this help message and exit
      -p, --plot           Plot result


interpolate
-----------

Interpolates an isochrone along a given parameter::

    usage: spexxytools isochrone interpolate [-h] [--count COUNT] [--column COLUMN] [--space SPACE SPACE] [-p] input output

    positional arguments:
      input                Input file containing the isochrone
      output               Output file

    optional arguments:
      -h, --help           show this help message and exit
      --count COUNT        Number of points for new isochrone
      --column COLUMN      Column used for interpolation
      --space SPACE SPACE  Space for distance calculations
      -p, --plot           Plot result
