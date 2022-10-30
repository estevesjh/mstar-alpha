import os
import gc
import atexit

from sys import exit

import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPRegressor, MLPClassifier

import joblib


from .plot import Plot
from .utils import load_cosmos_sample, create_directories
from .conf import set_ann_params

default_model_fname =  {'smass':'./data/model/ann', 'quenching': './data/model/classifier_ann'}


class Mstar:
    """
    Top-level class for training or loading the saved ANN model for stellar mass and quenching label predictions.
    The code is based on the the scikit-learn library and api.

    Parameters
    ----------
    kind: str, optional
        Specifies the prediction mode you will use. The string has to be "smass" or "quenching".

    model_fname: st, optional
        Sets the model filename to be loaded. Default filename is `./models/ann3`.
        
    outdir: bool, optional
        The path to save the estimated values, plots and the model.
    """

    def __init__(self, kind='smass', model_fname=None, rebuild=False, root='./data'):

        # Initialise arguments
        self.kind = kind
        self.outdir = outdir
        
        # default ANN model file
        if model_fname is None:
            model_fname = default_model_fname[self.kind]
            
        self.model_fname = model_fname
        
        # Define directory paths
        self.path = root
        self.plot_folder = root+'plots/'
        self.model_folder = root+'model/'

        # Check if model_name exists
        if os.path.isfile(self.model_folder+self.model_name) & (not rebuild):
            # load model here
            self.model = joblib.load(self.model_fname)
            print(f'Loaded model {self.model_name}')
        elif rebuild:
            print('Rebuilding model. It might take few seconds.')
            self.rebuild()
            
            print('Created default model file succsefully: {default_model_fname[self.kind]}')
            
        else:
            print('Model error: {self.model_fname} file not found. To run Mstar specify the correct model path/name.')
            print('If there is no {default_model_fname[self.kind]}, set rebuild to True')    
            exit()

    def predict(self, X):
        """Predicts the value of the model

        The default is to predict the stellar masses. 
        In the classifying a quenched galaxy returns the quenching probability. 
        
        Parameters
        ----------
        X : array
            An array of input features of galaxies with shape [n, x], where n is the number of galaxies and x is the number of input features.
            
        Returns
        ----------
        Y : array
            An array of predicted values with size n
        """
        # Use the model to make predictions on new objects
        y_pred = self.model.predict(self.x_test)

        # Update class variables
        self.plot.y_pred = y_pred
        
        if self.kind=='quenching':
            # quenching probability
            y_pred = y_pred[:,0]
        
        return y_pred
    
    def rebuild(self):
        self.path = './data/'
        self.model_fname = default_model_fname[self.kind]
        
        self.load_training_sample()
        self.fit(self.x_train, self.y_train, save_model=True)
    
    def fit(self, X, y, ann_params=None, save_model=True):
        """fit a new ANN model

        Fit a new ANN model. 
        You can set the best params with ann_params. 

        Parameters
        ----------
        X : array
            An array of input features of training galaxies with shape [n, x], where n is the number of galaxies and x is the number of input features.
        
        y : array
            An array of target features of training galaxies with shape [n, y], where n in the number of galaxies and y is the number of target features.

        ann_params: dict, optional
            If none sets the default ann_params. Otherwise, take a look at the sklearn api.
        
        save_model: bool, optional
            Whether to save model or not.
        """
        # Create the model directory
        create_directories(self.path)

        self.load_training_sample()
        
        if ann_params is None:
            # Get random forest hyperparameters
            self.params = set_ann_params(self.kind)

        func_dict = {'smass':MLPRegressor, 'quenching': MLPClassifier}
        func = func_dict[self.kind]
        
        # Train model
        print('Training model...')
        self.model = func(**self.params)
        self.model.fit(X, y)
                
        # Save model
        if save_model:
            joblib.dump(self.model, self.model_fname)
            print('Saved model')

        # # Initialise external classes
        # self.plot = Plot(y_test=self.y_test, y_pred=None, posteriors=None, target_features=self.target_features,
        #                  validation=None, no_samples=self.no_samples, no_features=self.no_features, path=self.path)

        # @atexit.register
        # def end():
        #     for obj in gc.get_objects():
        #         if isinstance(obj, h5py.File):
        #             obj.close()

    # External Class functions
    def load_training_sample(self):
        x_train, y_train, x_test, y_test = load_cosmos_sample(self.kind, './data/')
        
        # set training and test variables
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test
        pass
        
    def plot_scatter(self, show=False, save=True):
        print('Creating scatter plots...')
        return self.plot.plot_scatter(show=show, save=save)

    def plot_joint(self, show=False, save=True):
        print('Creating posterior plots...')
        return self.plot.plot_joint(show=show, save=save)