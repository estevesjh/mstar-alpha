import os
import gc
import atexit
from sys import exit
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
from .validation import Validation
from .plot import Plot
from .utils import convert_1d_arrays, create_directories
from .conf import set_rf_params


class Mstar:
    """
    Top-level class for training or loading the saved ann model.
    Uses the the scikit-learn library and api.

    Parameters
    ----------
    model_name: str
        The folder into which all the information is stored. To train a new model, specify a unique name or to load a
        previously trained model, specify its model name.

    x_train: array_like
        An array of input features of training galaxies with shape [n, x], where n is the number of galaxies and x is
        the number of input features.

    y_train: array_like
        An array of target features of training galaxies with shape [n, y], where n in the number of galaxies and y is
        the number of target features.

    x_test: array_like
        An array of input features of testing galaxies with the same shape as x_train.

    y_test: array_like, optional
        An array of target features of testing galaxies with the same shape as y_train.

    target_features: list, optional
        A list of variables of target features.

    save_model: bool, optional
        Whether to save model or not.
    """

    def __init__(self, model_name, x_train, y_train, x_test, y_test=None, target_features=None, save_model=False):

        # Initialise arguments
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_features = target_features
        self.save_model = save_model

        # Internal parameters
        self.no_samples = self.x_test.shape[0]

        try:
            self.no_features = self.y_train.shape[1]
        except IndexError:
            self.no_features = self.y_train.ndim

        # Define directory paths
        self.path = os.getcwd() + '/galpro/' + str(self.model_name) + '/'
        self.point_estimate_folder = 'point_estimates/'
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'
        self.plot_folder = 'plots/'

        # Creating target variable names
        if self.target_features is None:
            self.target_features = []
            for feature in np.arange(self.no_features):
                self.target_features.append('var_' + str(feature))

        # Convert 1d arrays
        if self.no_features == 1:
            self.y_train, self.y_test = convert_1d_arrays(self.y_train, self.y_test)

        # Check if model_name exists
        if os.path.isdir(self.path):
            print('Model directory exists.')

            # Try to load the model
            if os.path.isfile(self.path + str(self.model_name) + '.sav'):
                self.model = joblib.load(self.path + str(self.model_name) + '.sav')
                max_leafs, max_samples = np.load(self.path + 'dimensions.npy')
                self.trees = sparse.load_npz(self.path + 'trees.npz')
                self.trees = self.trees.toarray()
                self.trees = self.trees.reshape(self.model.n_estimators, max_leafs, max_samples)
                print('Loaded model.')
            else:
                print('No model found. Choose a different model_name or delete the directory.')
                exit()

        else:
            # Create the model directory
            create_directories(model_name=self.model_name)

            # Get random forest hyperparameters
            self.params = set_rf_params()

            # Train model
            print('Training model...')
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)

            # Build trees
            self.trees, max_leafs, max_samples = self._build_trees()

            # Save model
            if save_model:
                model_file = self.model_name + '.sav'
                joblib.dump(self.model, self.path + model_file)

                # Save trees
                trees_sparse = self.trees.reshape(self.model.n_estimators, max_leafs * max_samples)
                trees_sparse = sparse.csr_matrix(trees_sparse)
                sparse.save_npz(self.path + 'trees.npz', trees_sparse)

                # Save dimensions
                np.save(self.path + 'dimensions.npy', np.array([max_leafs, max_samples]))
                print('Saved model.')

        # Initialise external classes
        self.plot = Plot(y_test=self.y_test, y_pred=None, posteriors=None, target_features=self.target_features,
                         validation=None, no_samples=self.no_samples, no_features=self.no_features, path=self.path)
        self.validation = Validation(y_test=self.y_test, y_pred=None, posteriors=None, validation=None,
                                     target_features=self.target_features, no_samples=self.no_samples,
                                     no_features=self.no_features, path=self.path)

        @atexit.register
        def end():
            for obj in gc.get_objects():
                if isinstance(obj, h5py.File):
                    obj.close()

    def point_estimate(self, save_estimates=False, make_plots=False):
        """
        Make point predictions on test samples using the trained model.

        Parameters
        ----------
        save_estimates: bool, optional
            Whether to save point estimates or not.

        make_plots: bool, optional
            Whether to make scatter plots or not.
        """

        print('Generating point estimates...')

        # Use the model to make predictions on new objects
        y_pred = self.model.predict(self.x_test)

        # Update class variables
        self.plot.y_pred = y_pred

        # Save plots
        if make_plots:
            self.plot_scatter()

        # Save predictions
        if save_estimates:
            with h5py.File(self.path + self.point_estimate_folder + "point_estimates.h5", 'w') as f:
                f.create_dataset('point_estimates', data=y_pred)
            print('Saved point estimates. Any previously saved point estimates have been overwritten.')

        return y_pred

    # External Class functions
    def validate(self, save_validation=False, make_plots=False):
        """
        Validate univariate and/or multivariate posterior PDFs generated by the trained model.

        Parameters
        ----------
        save_validation: bool, optional
            Whether to save validation or not.
            
        make_plots: bool, optional
            Whether to make validation plots or not.
        """

        print('Performing validation...')

        # Run validation
        validation = self.validation.validate(save_validation=save_validation, make_plots=make_plots)

        # Update validation
        self.plot.validation = validation

        return validation

    def plot_scatter(self, show=False, save=True):
        print('Creating scatter plots...')
        return self.plot.plot_scatter(show=show, save=save)

    def plot_marginal(self, show=False, save=True):
        print('Creating posterior plots...')
        return self.plot.plot_marginal(show=show, save=save)

    def plot_joint(self, show=False, save=True):
        print('Creating posterior plots...')
        return self.plot.plot_joint(show=show, save=save)

    def plot_corner(self, show=False, save=True):
        print('Creating posterior plots...')
        return self.plot.plot_corner(show=show, save=save)

    def plot_pit(self, show=False, save=True):
        print('Creating PIT plots...')
        return self.plot.plot_pit(show=show, save=save)

    def plot_coppit(self, show=False, save=True):
        print('Creating copPIT plots...')
        return self.plot.plot_coppit(show=show, save=save)

    def plot_marginal_calibration(self, show=False, save=True):
        print('Creating marginal calibration plots...')
        return self.plot.plot_marginal_calibration(show=show, save=save)

    def plot_kendall_calibration(self, show=False, save=True):
        print('Creating kendall calibration plots...')
        return self.plot.plot_kendall_calibration(show=show, save=save)