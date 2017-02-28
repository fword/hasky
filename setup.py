import os

import setuptools
from setuptools import setup

__version__ = '0.1'

setup(name='deepiu',
      version=__version__,
      packages=setuptools.find_packages(exclude=["*.util", "*.util.*", "util.*", "util"]))
