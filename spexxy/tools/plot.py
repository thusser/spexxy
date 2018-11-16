import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from spexxy.data import FitsSpectrum


def add_parser(subparsers):
    # add parser
    parser = subparsers.add_parser('plot', help='Plot one or more spectrum')
    parser.add_argument('spectra', type=str, help='Spectrum to plot', nargs='+')
    parser.add_argument('-o', '--output', help='Save plot to PDF file', type=str)
    parser.add_argument('-r', '--results', help='Include results', action='store_true')
    parser.add_argument('--range', type=float, nargs=2, help='Wavelength range to plot')

    # argparse wrapper for plot
    def run(args):
        plot(**vars(args))
    parser.set_defaults(func=run)


def plot(spectra: list, output: str = None, results: bool = False, range: list = None, **kwargs):
    # if spectra not a list, make it a list
    if not isinstance(spectra, list):
        spectra = [spectra]

    # check output
    pdf = None
    if output:
        # want a PDF?
        if output.endswith('.pdf'):
            pdf = PdfPages(output)
        else:
            # if no pdf, we only allow for one spectrum to plot
            if len (spectra) > 1:
                raise ValueError('Plotting into a file other than a PDF works for a single spectrum only!')

    # loop spectra
    for filename in sorted(spectra):
        # init figure
        print(filename)

        # plot
        plot_spectrum(filename, results=results, wave_range=range)

        # show
        if output:
            pdf.savefig(papertype='a4', orientation='landscape')
            plt.close()
        else:
            plt.show()

    # close file
    if output:
        pdf.close()


def plot_spectrum(filename: str, results: bool = False, wave_range: list = None):
    try:
        with FitsSpectrum(filename) as fs:
            # get header
            hdr = fs.header

            # get spectrum
            spec = fs['NORMALIZED'] if 'NORMALIZED' in fs and results else fs.spectrum

            # and model
            model = fs.best_fit if results else None

            # get residuals
            residuals = fs.residuals if results else None

            # good pixels
            valid = fs.good_pixels

    except AttributeError:
        return None

    # create figure and axes
    if results:
        fig, (ax_spectrum, ax_residuals) = plt.subplots(figsize=(11.69, 8.27), nrows=2, sharex=True,
                                                        gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(hspace=0)
    else:
        fig, ax_spectrum = plt.subplots(figsize=(11.69, 8.27))
        ax_residuals = None

    # plot spectrum
    ax_spectrum.plot(spec.wave, spec.flux, ls="-", lw=1., c="k", marker="None")
    ax_spectrum.set_yticks(ax_spectrum.get_yticks()[1:-1])
    ax_spectrum.set_ylabel('Flux', fontsize=20)

    # best fit
    if model:
        ax_spectrum.plot(spec.wave, model, ls="-", lw=1., c="r", marker="None")

    # legend
    ax_spectrum.plot([0], [0], c='#F7977A')
    ax_spectrum.plot([0], [0], c='b')
    ax_spectrum.legend(["Observation", "Model", "Mask", "Residuals"] if results else ["Observation", "Mask"],
                       bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)

    # residuals
    if results:
        ax_residuals.plot(spec.wave, residuals, ls="-", lw=1., c="b", marker="None")
        ax_residuals.set_xlabel(r"$\mathrm{Wavelength}\ \lambda\,[\mathrm{\AA}]$", fontsize=20)
        ax_residuals.set_ylabel('Residuals', fontsize=20)
    else:
        ax_spectrum.set_xlabel(r"$\mathrm{Wavelength}\ \lambda\,[\mathrm{\AA}]$", fontsize=20)

    # xlim
    wave_min = wave_range[0] if wave_range else np.nanmin(spec.wave)
    wave_max = wave_range[1] if wave_range else np.nanmax(spec.wave)
    ax_spectrum.set_xlim(wave_min, wave_max)

    # ylim
    w = (spec.wave >= wave_min) & (spec.wave <= wave_max)
    if len(valid) == len(spec):
        w &= valid

    # get min/max of spectrum in range
    tmp = spec.flux[w]
    ax_spectrum.set_ylim((np.nanmin(tmp), np.nanmax(tmp)))

    # and for the residuals
    if results:
        tmp = residuals[w]
        ax_residuals.set_ylim((np.nanmin(tmp), np.nanmax(residuals)))

    # ticks and other stuff
    for ax in [ax_spectrum, ax_residuals]:
        if ax is not None:
            ax.minorticks_on()
            ax.grid()

            # good pixels
            start = None
            for x in range(len(spec.wave)):
                if valid[x] == 0 and start is None:
                    start = spec.wave[x]
                if valid[x] == 1 and start is not None:
                    # mark area
                    ax.axvspan(start - spec.wave_step / 2., spec.wave[x], color='#F7977A', zorder=10, alpha=0.5)
                    start = None

    # title
    fig.text(0.5, 0.97, fs.filename, ha='center', size=16)
