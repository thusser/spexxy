#!/usr/bin/env python3
from setuptools import setup, find_packages
from spexxy.version import VERSION

setup(
    name='spexxy',
    version=VERSION,
    description='spexxy spectrum fitting package',
    author='Tim-Oliver Husser',
    author_email='thusser@uni-goettingen.de',
    packages=find_packages(include=['spexxy', 'spexxy.*']),
    scripts=['bin/spexxy', 'bin/spexxytools'],
    include_package_data=True,
    requires=['scipy', 'numpy', 'astropy', 'pandas', 'lmfit']
)
