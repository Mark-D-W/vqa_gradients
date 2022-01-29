# Set up with:
# python3 setup.py install
# or
# python3 setup.py install --compile
# to compile python code using cython
import sys

from setuptools import setup, find_packages

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



# Choose if to compile .py files with cython
if "--compile" in sys.argv:
    COMPILE = True
    sys.argv.remove("--compile")
else:
    COMPILE = False



VERSION = '0.0.1'
DESCRIPTION = 'Gradients of variational quantum circuits.'
#LONG_DESCRIPTION = ''

# Setting up
setup(
    name="vqa_gradients",
    version=VERSION,
    author="Mark Walker",
    author_email="mark.damon.walker@gmail.com",
    description=DESCRIPTION,
    #long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["cython","numpy","scipy","matplotlib","numdifftools","pytest"],
    keywords=['python'],
    classifiers= [
        "Programming Language :: Python :: 3",
    ]
)



# Build the cython extension
# --------------------------
ext_modules=[
    Extension("vqa_gradients.Series", ["vqa_gradients/Series.py"]),
    Extension("vqa_gradients.Optimize", ["vqa_gradients/Optimise.py"]),
    Extension("vqa_gradients.misc_functions", ["vqa_gradients/misc_functions.py"]),
    ]

if COMPILE:
    setup(
        name="cy_ext",
        cmdclass = {'build_ext': build_ext},
        script_args = ['build_ext', '--inplace'],
        ext_modules=ext_modules
    )
