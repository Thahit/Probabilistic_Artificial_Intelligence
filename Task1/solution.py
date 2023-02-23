import os
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict

import math
import torch
import gpytorch

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 150  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
        
        self.likely = gpytorch.likelihoods.GaussianLikelihood()
        self.mean_weight=1
    
    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        test_features=torch.tensor(test_features, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predicitions=self.model(test_features)
            
            gp_mean = predicitions.mean.numpy()
            gp_std = np.sqrt(predicitions.variance.numpy())
            pred = gp_mean*(self.mean_weight)
        #print(gp_mean.dtype, gp_std.dtype, pred.dtype)

        return pred, gp_mean, gp_std
        
    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        train_GT=torch.tensor(train_GT, dtype=torch.float32)
        train_features = torch.tensor(train_features, dtype=torch.float32)
        print("x.shape: ", train_features.shape,
              "\ny.shape: ", train_GT.shape)
        
        self.model = ExactGPModel(train_features, train_GT, self.likely)
        self.model.train()
        self.likely.train()
        training_iter = 50000
        
        #print("parameters of the model: ", self.model.named_parameters())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        #loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.likely, self.model)#loss
        loss_func = lossF
        
        
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            
            output = self.model(train_features)
            #print(output)
            
            # Calc loss and backprop gradients
            loss = -loss_func(output, train_GT)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()

def lossF(output, train_GT):
    output=output.mean
    
    cost = (train_GT - output) ** 2
    weights = torch.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = output < train_GT
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (output >= 1.2*train_GT)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return -torch.mean(cost * weights)

    

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
    
    #smaller dataset
    train_features=train_features[:500,]
    train_GT=train_GT[:500]
    test_features=test_features[:100]
     
    #train_features = torch.linspace(0, 1, 100)
    #train_GT = torch.sin(train_features * (2 * math.pi)) + torch.randn(train_features.size()) * math.sqrt(0.04)    
    #test_features=torch.linspace(0, 1, 10)

    # Fit the model
    print('Fitting model')
    model = Model()
    
    model.fitting_model(train_GT, train_features)
    
    # Predict on the test features
    print('Predicting on test features')
    
    predictions = model.make_predictions(test_features)
    #print(predictions)
    
    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')

if __name__ == "__main__":
    main()
