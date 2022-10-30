
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'Stellar mass galaxy estimator'

# Setting up
setup(
    name='mstar-alpha',
    version=VERSION,
    install_requires=['numpy', 'scikit-learn>=0.22.1', 'joblib',
                      'matplotlib', 'seaborn>=0.10.1', 'scipy>=1.6.0', 'pandas'],
    url='https://mstar-alpha.readthedocs.io/en/',
    license='MIT License',
    author='Johnny Esteves',
    author_email='jesteves@umich.edu',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    project_urls={
                   "readthedocs": "https://mstar-alpha.readthedocs.io/",
                   "GitHub": "https://github.com/estevesjh/mstar-alpha",
                   "arXiv": "https://arxiv.org/"
                }
)