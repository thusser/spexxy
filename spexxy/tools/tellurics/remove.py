import logging
import os
import astropy.io.fits as pyfits
import shutil

from spexxy.data import SpectrumFits


def add_parser(subparsers):
    """Adds 'spexxy tellurics remove' command

    :param subparsers:  Subparser to attach new one to.
    """

    # init parser
    parser = subparsers.add_parser('remove', help='Removes tellurics from a spectrum.')
    parser.set_defaults(func=run)

    # arguments
    parser.add_argument('spectra', help='List of processed spectra',
                        type=str, nargs='+')
    parser.add_argument('-t', '--tellurics', help='File containing tellurics',
                        type=str)
    parser.add_argument('-d', '--dir', help='Output directory', type=str,
                        default=".")
    parser.add_argument('-l', '--snlimit', type=float, default=20.,
                        help='minimum SNR for using tellurics from file')


def run(args):
    """
    Removes tellurics from the given file from all given spectra.

    :param args:    argparse namespace with
                    .spectra:   List of files containing spectra
                                to correct.
                    .tellurics: File containing the tellurics. If not
                                given, each spectrum is corrected by its
                                own tellurics.
                    .dir        Output directory. If '.' (default),
                                input spectra are overwritten.
                    .snlimit:   Maximum S/N of spectrum to use tellurics
                                from file. For higher S/N the spectrum's
                                own tellurics is used.
        """

    # load tellurics
    tellurics = SpectrumFits(args.tellurics) \
        if args.tellurics is not None and os.path.exists(args.tellurics) \
        else None

    # loop files
    for i, spec in enumerate(args.spectra):
        logging.info("({0:d}/{1:d}) {2:s}..."
                     .format(i + 1, len(args.spectra), spec))

        # get signal to noise
        snr = pyfits.getval(spec, "HIERARCH SPECTRUM SNRATIO")
        logging.info("  Found a S/N of {0:.2f}.".format(snr))

        # use tellurics from fitted spectrum itself?
        own_tell = False
        if tellurics is None:
            logging.info("  No tellurics file given, using tellurics "
                         "from fitted spectrum.")
            own_tell = True
        if not own_tell and snr is None:
            logging.info("  No S/N ratio found in file, using tellurics "
                         "from fitted spectrum.")
            own_tell = True
        if not own_tell and args.snlimit is None:
            logging.info("  No S/N limit given, using tellurics "
                         "from fitted spectrum.")
            own_tell = True
        if not own_tell and snr > args.snlimit:
            logging.info("  S/N of spectrum larger than given S/N, using "
                         "tellurics from fitted spectrum.")
            own_tell = True

        # get filename of new file
        filename = os.path.abspath(args.dir + os.sep + os.path.basename(spec))

        # copy spectrum
        shutil.copyfile(spec, filename)

        # init tellurics
        tell = None

        # open file
        with pyfits.open(filename, mode='update') as fits:
            # try to load tellurics
            try:
                # load tellurics
                if own_tell:
                    tell = fits["TELLURICS"].data

                # delete extension
                del fits["TELLURICS"]
            except KeyError:
                logging.error("  Could not load tellurics from "
                              "fitted spectrum.")

            # no tellurics? set from file...
            if tell is None:
                if tellurics is None:
                    logging.error("  ERROR! Neither tellurics file given nor "
                                  "tellurics found in spectrum, skipping...")
                    continue
                tell = tellurics.flux

            # divide spectrum by tellurics
            fits["PRIMARY"].data /= tell

            # divide SIGMAs by tellurics
            try:
                fits["SIGMA"].data /= tell
            except KeyError:
                pass

            # flush
            fits.flush()
