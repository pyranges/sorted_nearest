from distutils.core import setup
from setuptools import find_packages, Extension, Command
from Cython.Build import cythonize

import os
import sys

from sorted_nearest import __version__


CLASSIFIERS = """Development Status :: 5 - Production/Stable
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Bio-Informatics"""

# split into lines and filter empty ones
CLASSIFIERS = CLASSIFIERS.splitlines()


# macros = [("CYTHON_TRACE", "1")]

# if macros:
#     from Cython.Compiler.Options import get_directive_defaults
#     directive_defaults = get_directive_defaults()
#     directive_defaults['linetrace'] = True
#     directive_defaults['binding'] = True


# extension sources
macros = []

extensions = [Extension("sorted_nearest.src.sorted_nearest", ["sorted_nearest/src/sorted_nearest.pyx"],
                        define_macros=macros)]


setup(
    name = "sorted_nearest",
    version=__version__,
    packages=find_packages(),
    ext_modules = cythonize(extensions),
    description = \
    'Find nearest interval.',
    long_description = __doc__,
    author = "Endre Bakken Stovner",
    author_email='endrebak85@gmail.com',
    url = 'https://github.com/endrebak/sorted_nearest',
    license = 'New BSD License',
    classifiers = CLASSIFIERS,
    package_data={'': ['*.pyx', '*.pxd', '*.h', '*.c']},
    include_dirs=["."],
)
