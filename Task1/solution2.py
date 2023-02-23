import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict



# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 150  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    #def __init__(self, kernel=None, mean_weight=1, std_weight=0):
    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        
        # inspired by https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/gaussian_process/kernels.py#L1423
        #rbf Kernel = ca 260 cost
        #self.kernel = kernel#Matern(length_scale=.3, nu=1)
        #self.mean_weight=mean_weight
        #self.std_weight=std_weight
        
        self.kernel = Matern(length_scale=1, nu=0.5)
        self.mean_weight=1.1
        self.std_weight=0
        self.model = GaussianProcessRegressor(kernel=self.kernel, 
                                              random_state=0)
    
    def get_params(self, deep):
        dct={"kernel":self.kernel,
            "mean_weight": self.mean_weight,
            "std_weight": self.std_weight
            }
        return dct
    
    def set_params(self,**params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        gp_mean, gp_std=self.model.predict(test_features, return_std=True)
        pred=gp_mean*(self.mean_weight+self.std_weight*gp_std)

        return pred, gp_mean, gp_std
    
    def predict(self, test_features: np.ndarray):
        return self.make_predictions(test_features)
        
    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        self.model.fit(train_features, train_GT)

    def fit(self, train_features: np.ndarray, train_GT: np.ndarray):
        return self.fitting_model(train_GT,train_features)
     
def eval(model, x ,y):
    pred,_,_= model.predict(x)   
    return -cost_function(y, pred)
    
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)

def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    '''
    from sklearn.model_selection import RandomizedSearchCV
    model=Model
    hyperparameter_grid={
        "kernel": [Matern(nu=1, length_scale=1),
                   Matern(nu=1, length_scale=.5),
                   Matern(nu=.5, length_scale=1),
                   Matern(nu=.5, length_scale=.5),
                   Matern(nu=1.5, length_scale=1),
                   Matern(nu=1.5, length_scale=.5),
                   Matern(nu=2, length_scale=1),
                   Matern(nu=2, length_scale=.5),
                   ],
        "mean_weight":[1.1, 1.2, 1.3, 1.4],
        "std_weight":[0,.5, 1, 1.5, 2]
    }
    #best = mean 1.2 std=.5 matern scale1 nu1, score ca -2500
    #best2 = {'std_weight': 2, 'mean_weight': 1.1, 'kernel': Matern(length_scale=1, nu=0.5)}
    
    clf = RandomizedSearchCV(model(), param_distributions=hyperparameter_grid, 
                             random_state=0, n_iter=50, scoring=eval, cv=4, verbose=2)
    clf.fit(train_features, train_GT)
    print(clf.best_estimator_)
    print(clf.best_params_)
    '''
    
    
    # Fit the model
    print('Fitting model')
    model = Model()
    
    model.fitting_model(train_GT, train_features)
    
    # Predict on the test features
    print('Predicting on test features')
    
    predictions = model.make_predictions(test_features)
    print(predictions)
    
    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')
    

if __name__ == "__main__":
    main()
