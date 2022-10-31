import sys
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier

# local folder
from .utils import get_project_root, load_cosmos_sample, create_directories
from .utils import write_model, load_model
from .conf import set_ann_params

# package root path
root = get_project_root()
default_model_fname =  {'smass':root+'/data/model/ann.pkl', 'quenching': root+'/data/model/classifier_ann.pkl'}

# todos:
# make plot functions
# test on python 2
# review api

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
        
    root: bool, optional
        The root path to load the models and the data. Also, it saves plots and new models.
    """

    def __init__(self, kind='smass', model_fname=None, rebuild=False, root=root):
        # Initialise arguments
        self.kind = kind
        
        # default ANN model file
        if model_fname is None:
            model_fname = default_model_fname[self.kind]
            
        self.model_fname = model_fname
        
        # Define directory paths
        self.path = root+'/data/'
        self.lib_folder = self.path+'/lib/'
        self.plot_folder = self.path+'plots/'
        self.model_folder = self.path+'model/'

        # Check if model_name exists
        if os.path.isfile(self.model_fname) & (not rebuild):
            # load model here
            self.model = load_model(self.model_fname)
            print('Loaded model {model}'.format(model=self.model_fname))
        elif rebuild:
            print('Rebuilding model. It might take few seconds.')
            self.rebuild()
            
            print('Created default model file succsefully: {dfname}'.format(dfname=default_model_fname[self.kind]))
            
        else:
            print('Model error: {fname} file not found. To run Mstar specify the correct model path/name.'.format(fname=self.model_fname))
            print('If there is no {dfname}, set rebuild to True'.format(dfname=default_model_fname[self.kind]))    
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
        #self.plot.y_pred = y_pred
        
#         if self.kind=='quenching':
#             # quenching probability
#             y_pred = y_pred[:,0]
        
        return y_pred
    
    def rebuild(self):
        """rebuild fitted model
        """
        # set default fname
        self.model_fname = default_model_fname[self.kind]
        # load training files
        self.load_training_sample()
        # fit data and save the model
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
            write_model(self.model, self.model_fname)
            print('Saved model')

    # External Class functions
    def load_training_sample(self):
        x_train, y_train = load_cosmos_sample('train', self.kind, self.lib_folder)
        # set training variables
        self.x_train = x_train
        self.y_train = y_train
        pass
        
    def load_test_sample(self):
        x_test, y_test = load_cosmos_sample('test', self.kind, self.lib_folder)
        self.x_test = x_test
        self.y_test = y_test
        pass

    def load_target_sample(self):
        x_test, y_test = load_cosmos_sample('target', self.kind, self.lib_folder)
        self.x_target = x_test
        self.y_tartet = y_test
        pass

    def plot_scatter(self, show=False, save=True):
        print('Creating scatter plots...')
        return self.plot.plot_scatter(show=show, save=save)

    def plot_joint(self, show=False, save=True):
        print('Creating posterior plots...')
        return self.plot.plot_joint(show=show, save=save)
