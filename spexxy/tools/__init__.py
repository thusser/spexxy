import os
import pkgutil
import importlib


def add_tree_node(parser, package, help, aliases=None):
    # package name
    name = package.__name__
    cmd = name[name.rfind('.')+1:]

    # aliases
    if aliases is None:
        aliases = []

    # add parser
    p = parser.add_parser(cmd, help=help, aliases=aliases)
    sp = p.add_subparsers(help=help)

    # list modules in gcdb.methods.fits
    pkgpath = os.path.dirname(package.__file__)
    modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]

    # loop modules
    for m in modules:
        # import module
        mod = importlib.import_module(name + '.' + m)

        # add subparser
        mod.add_parser(sp)
