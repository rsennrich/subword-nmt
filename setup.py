from setuptools import setup, find_packages
import unittest


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('subword_nmt/tests', pattern='test_*.py')

    return test_suite


setup(
    name='subword_nmt',
    version='0.3',
    description='Subword Neural Machine Translation',
    url='https://github.com/rsennrich/subword-nmt',
    author='Rico Sennrich',
    license='MIT',
    test_suite='setup.test_suite',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['subword-nmt=subword_nmt.command_line:main'],
    },
    include_package_data=True
)
