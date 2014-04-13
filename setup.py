#! /usr/bin/env python

from distutils.core import setup

import os

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename), 'r').read()

setup(
    name='Sonus',
    version='1.0.1',
    description='Language Detection System in python',
    author='Bhuvan Anand, Krishna Ramesh',
    packages=['sonus', 'sonus.feature', 'sonus.utils', 'sonus.gmm'],
    long_description = read('README'),
    classifiers=[
        'Development Status :: 1.0.1',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: Linux',
        'Operating System :: Microsoft :: Microsoft Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python 2.7'
        ]
    )
