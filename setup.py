# Set up with python setup.py sdist bdist_wheel

from setuptools import setup, find_packages

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
    install_requires=["numpy","scipy","matplotlib"],         
    keywords=['python'],
    classifiers= [
        "Programming Language :: Python :: 3",
    ]
)




# Build the f2py fortran extension
# --------------------------------
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

flib = Extension(name = 'functions.flib',
                 #extra_compile_args = ['-O3'],
                 sources = ['src_fortran/functions.f90']
                 )

setup(ext_modules = [flib])
