from sys import exit
import os
import h5py
import numpy as np
from scipy.stats import entropy, kstest, median_abs_deviation, cramervonmises


def create_directories(path):
    # Define directories
    model_folder = 'model/'
    plot_folder = 'plots/'

    # Create model directory
    os.mkdir(path, exist_ok = True)
    os.mkdir(path + model_folder, exist_ok = True)
    os.mkdir(path + plot_folder, exist_ok = True)

def convert_1d_arrays(*arrays):
    """Convert given 1d arrays from shape (n,) to (n, 1) for compatibility with code."""

    arrays = list(arrays)
    for i in np.arange(len(arrays)):
        array = arrays[i]
        if array is not None:
            arrays[i] = arrays[i].reshape(-1, 1)

    return arrays

def load_cosmos_sample(kind='smass', path='./data/'):
    """Loads COSMOS training/test data."""
    x_train = np.load(path+'x_train.npy')
    y_train = np.load(path+'y_train.npy')

    x_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')

    ix = np.where(kind=='smass', 1, 0)
    
    return x_train, y_train[:,ix], x_test, y_test[:,ix]

def load_posteriors(path):
    """Loads saved posteriors."""

    posterior_folder = 'posteriors/'
    if os.path.isfile(path + posterior_folder + 'posteriors.h5'):
        posteriors = h5py.File(path + posterior_folder + "posteriors.h5", "r")
        print('Previously saved posteriors have been loaded.')
    else:
        print('No posteriors have been found. Run posterior() to generate posteriors.')
        exit()

    return posteriors