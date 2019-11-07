#!/usr/bin/env python
"""
PyNVT (Python Neuroimaging Visualization Tools)
"""
from distutils.core import setup
from setuptools import find_packages
import re, io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('pynvt/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

__author__ = 'SungHo Lee, Woomi Ban'
__email__ = 'shlee@unc.edu, banwoomi@email.unc.edu'

setup(name='PyNVT',
      version=__version__,
      description='Python Neuroimaging Visualization Tools',
      author=__author__,
      author_email=__email__,
      url=None,
      license='GNLv3',
      packages=find_packages(),
      install_requires=['pandas',
                        'matplotlib',
                        'scipy',
                        'scikit-image',
                        'sklearn',
                        'nibabel',
                        'numpy'
                       ],
      scripts=['PyNVT/pynvt',
              ],
      classifiers=[
            # How mature is this project? Common values are
            #  3 - Alpha
            #  4 - Beta
            #  5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Framework :: Jupyter',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Natural Language :: English',

            # Specify the Python version you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.7',
      ],
      keywords='Python Neuroimaging Visualization Tools'
      )
