from setuptools import setup, find_packages
import unittest
import codecs

def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('subword_nmt/tests', pattern='test_*.py')

    return test_suite


setup(
    name='subword_nmt',
    version='0.3.7',
    description='Unsupervised Word Segmentation for Neural Machine Translation and Text Generation',
    long_description=(codecs.open("README.md", encoding='utf-8').read() +
                      "\n\n" + codecs.open("CHANGELOG.md", encoding='utf-8').read()),
    long_description_content_type="text/markdown",
    url='https://github.com/rsennrich/subword-nmt',
    author='Rico Sennrich',
    license='MIT',
    test_suite='setup.test_suite',
    classifiers=[
    'Intended Audience :: Developers',
    'Topic :: Text Processing',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': ['subword-nmt=subword_nmt.subword_nmt:main'],
    },
    include_package_data=True
)
