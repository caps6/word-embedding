#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path as os_path

# Package meta-data.
NAME = 'pyembed'
DESCRIPTION = 'Embed your own words (with word2vec).'
URL = 'https://github.com/acapitanelli/word-embedding'
AUTHOR = 'Andrea Capitanelli'
EMAIL = 'andrea@capitanelli.gmail.com'
VERSION = '1.0.0'

# short/long description
here = os_path.abspath(os_path.dirname(__file__))
try:
    with open(os_path.join(here,'README.md'),'r',encoding='utf-8') as f:
        long_desc = '\n' + f.read()
except FileNotFoundError:
    long_desc = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    url=URL,
    python_requires='>=3.6.0',
    packages=find_packages(),
    long_description=long_desc,
    long_description_content_type='text/markdown',
    keywords='word embedding neural nets',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    scripts=[
        os_path.join(here,'bin/pyembed')
    ]
)
