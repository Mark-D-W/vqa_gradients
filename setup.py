# Set up with python setup.py sdist bdist_wheel

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Gradients of variational quantum circuits.'
#LONG_DESCRIPTION = ''

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
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
