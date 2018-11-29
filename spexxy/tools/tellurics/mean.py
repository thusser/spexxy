import glob
import logging
import numpy as np
from spexxy.utils.fits import bulk_read_header

from spexxy.data import FitsSpectrum, SpectrumFits


def add_parser(subparsers):
    """
    Adds 'spexxy tellurics mean' command

    :param subparsers:  Subparser to attach new one to.
    """

    # init parser
    parser = subparsers.add_parser('mean', help='Calculates the mean tellurics from a list of fits.')
    parser.set_defaults(func=run)

    # calculate mean
    parser.add_argument('spectra', help='List of processed spectra', type=str, nargs='+')
    parser.add_argument('-o', '--output', help='File to write tellurics to', type=str, default="tellurics.fits")
    parser.add_argument('-l', '--snlimit', help='minimum SNR to use', type=float)
    parser.add_argument('-f', '--snfrac', help='if snlimit is not given, calculate it from X% best',
                        type=float, default=10)
    parser.add_argument('-w', '--weight', help='weight by SNR', action="store_true")
    parser.add_argument('-r', '--resample', help='resample to start,step,count', nargs=3,
                        default=(None, None, None), type=float)


def run(args):
    """
    Takes a list of spectra and calculates the mean of the fitted tellurics.

    :param args:    argparse namespace with
                    .spectra:   List of files containing spectra.
                    .output:    Output file for mean tellurics
                                (default: tellurics.fits)
                    .snlimit:   Only calculate mean tellurics for spectra
                                with a S/N higher than the given number.
                    .weight:    If set, tellurics are weightes by the S/N
                                of their spectra.
    """

    # get all spectra
    filenames = []
    for s in args.spectra:
        if '*' in s:
            filenames.extend(glob.glob(s))
        else:
            filenames.append(s)

    # read headers
    logging.info('Reading all FITS headers...')
    headers = bulk_read_header(filenames, ['HIERARCH SPECTRUM SNRATIO'])

    # get all snr values
    logging.info('Extracting all S/N values...')
    snrs = {}
    for i, row in headers.iterrows():
        try:
            snrs[row['FILE']] = float(row['HIERARCH SPECTRUM SNRATIO'])
        except ValueError:
            continue

    # no snlimit given?
    if args.snlimit is not None:
        snlimit = args.snlimit
    else:
        tmp = sorted(snrs.values())
        logging.info('Calculating S/N limit from %.2f%% highest S/N values...', args.snfrac)
        snlimit = tmp[-int(len(tmp)*args.snfrac/100)]
    logging.info('Using S/N limit of %.2f...', snlimit)

    # filter by snlimit
    logging.info('Filtering spectra by S/N limit...')
    spectra = [filename for filename, snr in snrs.items() if snr > snlimit]

    # tellurics spectrum
    tellurics = None
    weights = None
    count = None
    wave_start, wave_step, wave_count = args.resample
    if wave_count is not None:
        wave_count = int(wave_count)

    # loop files
    logging.info('Processing %d spectra...', len(spectra))
    for i, spec in enumerate(spectra, 1):
        # open file
        with FitsSpectrum(spec, 'r') as fs:
            # get signal to noise
            snr = fs.header["HIERARCH SPECTRUM SNRATIO"]

            # print
            logging.info('(%d/%d) %s %-5.2f', i, len(spectra), spec, snr)

            # weight
            weight = snr if args.weight else 1.

            # some more info
            if wave_start is None:
                # WAVE extension or CRVAL/CDELT?
                if 'CRVAL1' in fs.header and 'CDELT1' in fs.header:
                    wave_start = fs.header["CRVAL1"]
                    wave_step = fs.header["CDELT1"]
                    if fs.header['CUNIT1'] == 'm':
                        wave_start *= 1e10
                        wave_step *= 1e10
                    wave_count = fs.header['NAXIS1']
                elif 'WAVE' in fs.header and fs.header['WAVE'] in fs:
                    logging.error('Combining tellurics on PIXTABLE spectra not allowed without resampling.')
                    continue
                else:
                    logging.error('Could not determine wavelength grid.')
                    continue

            # get tellurics
            tell = fs.tellurics
            if not tell:
                continue

            # resample
            tell = tell.resample_with_holes(wave_start=wave_start, wave_step=wave_step, wave_count=wave_count)

            # tellurics exist? on first iteration we create the array.
            if tellurics is None:
                tellurics = np.zeros((wave_count))
                weights = np.zeros((wave_count))
                count = np.zeros((wave_count))

            # add to sum
            w = np.where(~np.isnan(tell.flux))
            tellurics[w] += weight * tell.flux[w]
            weights[w] += weight
            count[w] += 1

    # divide by sum
    w = np.where(~np.isnan(tellurics) & ~np.isnan(weights))
    tellurics[w] /= weights[w]

    # create spectrum for tellurics and save it
    tell_spec = SpectrumFits.from_flux(tellurics, wave_start, wave_step, primary=True)
    tell_spec.save(args.output)

    # output
    logging.info("Finished successfully.")
