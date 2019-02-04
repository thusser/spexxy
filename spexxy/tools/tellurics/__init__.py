import sys
from spexxy.tools import add_tree_node


def add_parser(parser):
    add_tree_node(parser, sys.modules[__name__], 'tellurics operations')
