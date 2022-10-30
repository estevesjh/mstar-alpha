from setuptools import setup

setup(
    name='mstar-alpha',
    version='0.0.1',
    packages=['mstar'],
    install_requires=['numpy', 'scikit-learn>=0.22.1', 'joblib', 'h5py',
                      'matplotlib', 'statsmodels==0.11.0', 'seaborn==0.10.1', 'scipy>=1.6.0', 'pandas'],
    url='https://mstar-alpha.readthedocs.io/en/',
    license='MIT License',
    author='Johnny Esteves',
    author_email='jesteves@umich.edu',
    description='Stellar mass galaxy estimator',
    project_urls={
        "readthedocs": "https://mstar-alpha.readthedocs.io/",
        "GitHub": "https://github.com/estevesjh/mstar-alpha",
        "arXiv": "https://arxiv.org/"
    }
)
