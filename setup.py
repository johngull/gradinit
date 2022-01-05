from distutils.core import setup

readme = open('README.md').read()

VERSION = '0.1.0'

requirements = [
    'torch',
]

setup(
    name='gradinit',
    version=VERSION,
    packages=['gradinit'],
    author='Vitaly Bondar',
    author_email='johngull@gmail.com',
    license='MIT',
    description='Pytorch implementation of the gradient-based initialization',
    long_description=readme,
    long_description_content_type='text/markdown',

    install_requires=requirements,
)