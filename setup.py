from setuptools import setup

VERSION = '0.1.1'

requirements = [
    'torch',
]

setup(
    name='gradinit',
    version=VERSION,
    packages=['gradinit'],
    url='https://github.com/johngull/gradinit',
    author='Vitaly Bondar',
    author_email='johngull@gmail.com',
    license='MIT',
    description='Pytorch implementation of the gradient-based initialization',

    install_requires=requirements,
)
