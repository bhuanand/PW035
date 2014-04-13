#! /usr/bin/env python

from distutils.core import setup

setup(
    name='Sonus',
    version='1.0',
    description='Language Detection System in python',
    author='Bhuvan Anand, Krishna Ramesh',
    packages=['sonus', 'sonus.feature', 'sonus.utils', 'sonus.gmm'],
    classifiers=[
        'Development Status :: 1.0',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: Linux',
        'Operating System :: Microsoft :: Microsoft Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python 2.7'
        ]
    )
