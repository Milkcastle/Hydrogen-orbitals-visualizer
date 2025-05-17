# setup.py
from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='my_orbital_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sympy',
        'matplotlib',
        'scipy',
    ],
    author='Your Name',
    description='Hydrogen atom orbital visualizer and hybrid orbital constructor',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)