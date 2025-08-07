# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("CyFunctions", ["CyFunctions.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("Calculators.Probabilities", ["Calculators/Probabilities.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="Run baby run",
    ext_modules=cythonize(extensions),
)

# setup(
#     ext_modules=cythonize(
#         "KM3NeT_Cython.pyx", compiler_directives={"language_level": "3"}
#     )
# )