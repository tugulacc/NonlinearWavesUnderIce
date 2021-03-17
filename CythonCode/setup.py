from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(['runCode.pyx','Preconditioner.pyx','plotSurface.pyx','icefuncs.pyx','BIFunction.pyx','auxfuncs.pyx']),include_dirs=[numpy.get_include()])