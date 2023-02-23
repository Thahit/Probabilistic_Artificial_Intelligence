import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.theta = []
        self.f = []
        self.v = []

        self.sigma_f = 0.15
        self.var_f = self.sigma_f**2

        self.sigma_v = 0.0001
        self.var_v = self.sigma_v**2

        self.v_threshold = 1.2
        self.beta = 1

        self.f_kernel = 0.5*Matern(length_scale=0.5, nu=2.5)
        self.model_f = GaussianProcessRegressor(
            kernel=self.f_kernel, 
            alpha=self.var_f, 
            optimizer=None, 
            normalize_y=False,
            random_state=42)

        self.v_kernel = ConstantKernel(1.5) + ((2)**0.5)*Matern(length_scale=0.5, nu=2.5)
        self.model_v = GaussianProcessRegressor(
            kernel=self.v_kernel, 
            alpha=self.var_v,
            optimizer=None,
            normalize_y=False,
            random_state=42)


    def get_data(self):
        """
        Return data in correctly formatted numpy array.

        Returns
        -------
        np.ndarray
            length x domain.shape[0] array
        """
        theta = np.array(self.theta).reshape(len(self.theta),-1)
        f = np.array(self.f).reshape(len(self.f),-1)
        v = np.array(self.v).reshape(len(self.v), -1)
        return theta, f, v


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        
        assert (len(self.theta) != 0), f"no initial guess for parameter theta"

        # Fit GP
        theta, f, v = self.get_data()
        self.model_f.fit(X=theta, y=f)
        self.model_v.fit(X=theta, y=v)
        
        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []  

        # f_values calculated by:
        # 

        x_values = []  

        # Restarts the optimization 20 times and pick best solution
        for _ in range(200):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0]) 
            #x0 has shape (1,) i.e. a 1D array
            #x0 are points randomly drawn from domain where we want to evaluate acquisition function

            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)

            # result[0] is location in domain
            # result[1] is value of acquisition function

            x_values.append(np.clip(result[0], *domain[0]))

            f_values.append(-result[1]) 

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        mu_f, std_f = self.model_f.predict(x.reshape(1,-1), return_std=True)
        mu_v, std_v = self.model_v.predict(x.reshape(1,-1), return_std=True)

        if mu_v[0] - 3*std_v[0] < self.v_threshold: 
            # i.e. if we violate speed constraint with high probability at given theta
            # return a terrible score (Q: does 0 make sense?)
            return float(0.0)
        else: 
            return float(mu_f[0] + self.beta*std_f[0])
            # for points that have low probability of violating speed constraint
            # calculate acquisition score (exploitation vs exploration)


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.theta.append(x)
        self.f.append(f)
        self.v.append(v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        theta, f, v = self.get_data()
        theta = theta[v > self.v_threshold]
        f = f[v > self.v_threshold]  
        v = v[v > self.v_threshold]
        idx = np.argmax(f)
        print("valid accuracy of chosen model: {} \t speed of chosen model: {}".format(f[idx], v[idx]))      



        return theta[idx]
""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()
    n_dim = domain.shape[0]

    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()