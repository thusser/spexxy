import sys
from spexxy.utils import methods


def add_parser(parser):
    methods.add_tree_node(parser, sys.modules[__name__], 'Tellurics operations')
