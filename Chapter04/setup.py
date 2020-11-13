from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['gcsfs==0.5.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Pandas library to read data from files stored in GCS.'
)
