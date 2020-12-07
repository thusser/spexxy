from spexxy.data.lsf import LSF
from spexxy.data.spectrum import Spectrum


def add_parser(subparsers):
    # apply
    parser = subparsers.add_parser('apply')
    parser.set_defaults(func=run)
    parser.add_argument('lsf', metavar='lsf', type=str, help='LSF to convolve with')
    parser.add_argument('spectrum', metavar='spectrum', type=str, help='spectrum to convolve')
    parser.add_argument('output', metavar='output', type=str, help='output file')


def run(args):
    # load LSF and spec
    lsf = LSF.load(args.lsf)
    spec = Spectrum.load(args.spectrum)

    # scale
    lsf.resample(spec)

    # apply
    output = lsf(spec)

    # save it
    output.save(args.output)
