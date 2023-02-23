import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
import matplotlib.pyplot as plt
from copy import deepcopy

domain = np.array([[0, 5]])

""" Solution """
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

class BO_algo():
    """
    My solution implements the SafeOPT algorithm, based on the following paper: https://arxiv.org/abs/1509.01066
    """

    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # DONE: enter your code here

        kernel_f = 0.5 * Matern(length_scale=0.5, nu=2.5)
        kernel_v = ConstantKernel(constant_value=1.5) + (2 ** 0.5) * Matern(length_scale=0.5, nu=2.5)
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f, random_state=0, optimizer=None, alpha=0.15**2)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v, random_state=0, optimizer=None, alpha=0.0001**2)
        self.min_v = 1.2
        self.beta = 2

        self.datapoints = []
        self.A = set(np.linspace(0.0, 5.0, num=1500))
        self.S = set()
        self.M = set()
        self.G = set()

        self.i = 0
        self.n_random_samples = 3

    def setup_sets(self):
        # Initialize the confidence sets S, M, G
        A = list(self.A)                                                        # All inspected datapoints in the domain ([0, 5] discretized)
        YAF = self.gp_f.predict(np.array(A).reshape(-1, 1), return_std=True)    # GP posterior of A modeling the accuracy
        YAV = self.gp_v.predict(np.array(A).reshape(-1, 1), return_std=True)    # GP posterior of A modeling the speed
        LAV = YAV[0].flatten() - self.beta * YAV[1].flatten()                   # lower confidence bound of A on speed

        # safe elements S:= all a in A that has a higher lower confidence bound than the threshold according to the GP
        for i in range(len(A)):
            if LAV[i] >= self.min_v:
                self.S.add(A[i])

        S = list(self.S)
        if S:
            YS = self.gp_f.predict(np.array(S).reshape(-1, 1), return_std=True) # GP posterior of sc modeling the accuracy
            US = YS[0].flatten() + self.beta * YS[1].flatten()                  # upper bound in S
            LS = YS[0].flatten() + self.beta * YS[1].flatten()                  # lower bound in S

            # maximizers M:= all elements in S that have a higher upper bound the highest lowest bound in S
            for i in range(len(S)):
                if US[i] >= np.max(LS):
                    self.M.add(S[i])

        # expanders G:= all elements in S \ M that if they were chosen other A \ S elements would be considered safe
        M = list(self.M)
        if M:
            A_S = list(self.A.difference(self.S))
            S_M = list(self.S.difference(self.M))
            if S_M:
                YM = self.gp_f.predict(np.array(M).reshape(-1, 1), return_std=True)
                WM = 2 * self.beta * YM[1].flatten()
                YS_M = self.gp_f.predict(np.array(S_M).reshape(-1, 1), return_std=True)
                WSM = 2 * self.beta * YS_M[1].flatten()

                # Sort in descending order based on w, therefore we can break when we find the first expander
                SM_sorted = [(a, b) for a, b in sorted(zip(S_M, WSM), key=lambda pair: -pair[1])]
                # Iterate through on all s_ in S\M with a higher unconfidence (w) than the least confident sample in M
                SM_sorted = [a for a, b in list(filter(lambda t: True if t[1] > np.max(WM) else False, SM_sorted))]

                b = False
                for s_ in SM_sorted:
                    if b:
                        break

                    # Simulate if 's_' was selected with its upper bound on V
                    mu_v, sigma_v = self.gp_v.predict(np.array(s_).reshape(1, -1), return_std=True)
                    u_v = mu_v + self.beta * sigma_v
                    gpv_fake = deepcopy(self.gp_v)
                    gpv_fake.fit(np.array(s_).reshape(1, -1), u_v)

                    # Check if there are any new samples in A\S that could be considered safe
                    for a_ in A_S:
                        fake_v_m, fake_v_s = gpv_fake.predict(np.array(a_).reshape(1, -1), return_std=True)
                        if fake_v_m - self.beta * fake_v_s >= self.min_v:
                            self.G.add(s_)
                            b = True
                            break

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # DONE: enter your code here
        self.setup_sets()

        result = None
        # For the first few iteration sample randomly or until the G union M set is not empty
        if self.i < self.n_random_samples or len(self.M) + len(self.G) < 1:
            result = np.random.uniform(low=domain[0][0], high=domain[0][1])
        # Recommend c in G union M with the largest w (u - l)
        else:
            candidates = list(self.M.union(self.G))
            m, s = self.gp_f.predict(np.array(candidates).reshape(-1, 1), return_std=True)
            result = candidates[np.argmax(2 * self.beta * s)]

        self.i += 1

        return np.array([result]).reshape((1, -1))

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        raise NotImplementedError('Should not be called')
        pass

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

        raise NotImplementedError('Should not be called')
        pass

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

        # DONE: enter your code here
        if f is not np.ndarray:
            f = np.array(f).reshape(1, -1)
        if v is not np.ndarray:
            v = np.array(v).reshape(1, -1)

        # Fit the gps
        X = np.array([d[0] for d in self.datapoints]).reshape(-1, 1)
        X = np.concatenate((X, x))
        F = np.array([d[1] for d in self.datapoints]).reshape(-1, 1)
        F = np.concatenate((F, f))
        V = np.array([d[2] for d in self.datapoints]).reshape(-1, 1)
        V = np.concatenate((V, v))

        self.gp_f.fit(X, F)
        self.gp_v.fit(X, V)

        # store the tuple
        self.datapoints.append((x, f, v))

        # clear S
        self.S = set()
        #self.G = set()
        #self.M = set()

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # DONE: enter your code here

        # if there are no maximizers
        if len(self.M) == 0:
            dps = filter(lambda t: True if t[2] >= self.min_v else False, self.datapoints)
            # return the datapoint with the best function value, that holds the safety constraint
            if len(list(dps)) > 1:
                return max(dps, key=lambda t: t[1])[0]
            # if there is no such point, return the best safe sample
            else:
                S_Y = self.gp_f.predict(np.array(list(self.S)).reshape(-1, 1))
                ss = list(self.M)[np.argmax(S_Y)]
                return ss
        else:
            # return the best maximizer
            maximizer_Y = self.gp_f.predict(np.array(list(self.M)).reshape(-1, 1))
            ms = list(self.M)[np.argmax(maximizer_Y)]
            return ms

        # note that the program can fail here, if M, S, and D that holds the safety constraint are empty.
        # that means we have an ill-configurated model, try tuning the parameters.


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

    # Visualizing
    plt.plot([x[0].flatten() for x in agent.datapoints], [x[1].flatten() for x in agent.datapoints], label = 'learned points')
    df = np.linspace(0, 5, num=20)
    plt.plot(df, [f(x_) for x_ in df], label = 'fx')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()