from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required_pkgs = f.read().splitlines()

setup(
    name='autorecipe',
    version='0.1',
    packages=find_packages(),
    install_requires=required_pkgs,
)
