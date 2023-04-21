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
    entry_points={
        'console_scripts': [
            'spexxy=spexxy.cli.spexxy:main',
            'spexxytools=spexxy.cli.spexxytools:main',
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'numpy',
        'astropy',
        'pandas',
        'lmfit',
        'pyyaml',
        'matplotlib',
        'pyyaml',
        'h5py',
        'PyQt5'
    ]
)
