from setuptools import setup, find_packages

setup(

    name="quad_theano",
    packages=find_packages(),
    version='v0.1',
    description='theano integration routines',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
    requires=[
        'numpy',
        'thenao',
        
    ]

)
