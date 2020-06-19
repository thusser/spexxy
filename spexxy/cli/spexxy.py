import yaml
import argparse
import logging
import sys
from collections import OrderedDict

from spexxy.application import Application
from spexxy.utils.log import setup_log


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Work-around for loading the config YAML file into an OrderedDict.
    Taken from: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    Should not be required in Python 3.7+, where a dict is always sorted."""
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, Loader=OrderedLoader)


def main():
    # init logging
    logging.getLogger().setLevel(logging.DEBUG)
    setup_log('spexxy.main', 'spexxy.log', mode='w')

    # init parser
    parser = argparse.ArgumentParser(description='spexxy spectrum fitting tool')
    parser.add_argument('config', help='spexxy configuration file', type=argparse.FileType('r'))
    parser.add_argument('filenames', help='filenames of spectra to fit', type=str, nargs='*')
    parser.add_argument('--mp', type=int, help='run fit in parallel in the given number of processes')
    parser.add_argument('-o', '--output', help='CSV to write results into', type=str, default='spexxy.csv')
    parser.add_argument('-r', '--resume', help='resume previous fit', action='store_true')

    # parse args
    args = parser.parse_args()

    # parse config, use ordered_load, if version<3.7
    if sys.version_info >= (3, 7):
        config = yaml.load(args.config, Loader=yaml.FullLoader)
    else:
        config = ordered_load(args.config)

    # create app and run it
    app = Application(config, filenames=args.filenames, ncpus=args.mp, output=args.output, resume=args.resume)
    app.run()


if __name__ == '__main__':
    main()
