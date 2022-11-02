from sys import exit
import os
import pickle
import numpy as np
from pathlib import Path

def get_project_root():
    return str(Path(__file__).absolute().parent.parent)

def load_cosmos_sample(label,kind='smass', path='./data/lib/'):
    """Loads COSMOS training/test data."""
    x_train = np.load(path+'x_%s.npy'%label)
    y_train = np.load(path+'y_%s.npy'%label)

    ix = np.where(kind=='smass', 0, 1)
    
    return x_train, y_train[:,ix]

def write_model(model, fname):
    with open(fname, 'wb') as f:
        d = pickle.dump(model, f)
    pass

def load_model(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='bytes') 
    return d

def create_directories(path):
    # Define directories
    model_folder = 'model/'
    plot_folder = 'plots/'

    # Create model directory
    _makedirs(path)
    _makedirs(path + model_folder)
    _makedirs(path + plot_folder)

def _makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    pass

def convert_1d_arrays(*arrays):
    """Convert given 1d arrays from shape (n,) to (n, 1) for compatibility with code."""

    arrays = list(arrays)
    for i in np.arange(len(arrays)):
        array = arrays[i]
        if array is not None:
            arrays[i] = arrays[i].reshape(-1, 1)

    return arrays

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