import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
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
        # load spectrum
        with FitsSpectrum(filename) as fs:
            # get spectrum
            spec = fs['NORMALIZED'] if 'NORMALIZED' in fs and results else fs.spectrum

            # and model
            model = fs.best_fit if results else None

            # get residuals
            residuals = fs.residuals if results else None

            # good pixels
            valid = fs.good_pixels

        # plot
        plot_spectrum(spec, model, residuals, valid, wave_range=range, title=fs.filename)

        # show
        if output:
            pdf.savefig(papertype='a4', orientation='landscape')
            plt.close()
        else:
            plt.show()

    # close file
    if output:
        pdf.close()


def plot_spectrum(spec, model: 'Spectrum' = None, residuals: np.ndarray = None, valid: np.ndarray = None,
                  wave_range: list = None, text: str = None, text_width: float = 0.3, title: str = None):
    # specify grid
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
    gs.update(wspace=0., hspace=0., left=0.09, right=0.99, top=0.95, bottom=0.08)

    # adjust top to allow for a title
    if title is not None:
        gs.update(top=0.92)

    # adjust right to allow for some text
    if text is not None:
        gs.update(right=1.0 - text_width)

    # create figure and axes
    if residuals is None:
        ax_spectrum = plt.subplot(gs[:, 0])
        ax_residuals = None
    else:
        ax_spectrum = plt.subplot(gs[0, 0])
        ax_residuals = plt.subplot(gs[1, 0], sharex=ax_spectrum)

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
    labels = ["Observation", "Model", "Mask", "Residuals"] if ax_residuals is not None else ["Observation", "Mask"]
    ax_spectrum.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

    # residuals
    if ax_residuals is not None:
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
    if ax_residuals is not None:
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
    if title is not None:
        fig.text(0.5, 0.97, title, ha='center', size=16)

    # add text
    fig.text(1.0 - text_width + 0.02, 0.9, text, family='monospace', va='top', ha='left')

    # finished
    return fig
