#acquisitions.py
from graphlearning.active_learning import acquisition_function
import numpy as np


class uncnormprop_plusplus(acquisition_function):
    '''
    Sample proportional to the acquisition function's values.
    
    Currently implemented for SEQUENTIAL only
    '''
    def __init__(self):
        self.K = 10
        self.log_Eps_tilde = np.log(1e150)  # log of square root of roughly the max precision of python float
    
    def set_K(self, K):
        print(f"Setting K = {K} for uncnormprop++")
        self.K = K
        
    def compute(self, u, candidate_ind):
        vals = 1. - np.linalg.norm(u[candidate_ind, :], axis=1)

        M = vals.max()
        T0 = M - np.percentile(vals, 100 * (1. - 1. / self.K))
        eps = M / (self.log_Eps_tilde - np.log(vals.size))
        T = max(eps, min(1.0, T0))
        p = np.exp(vals / T)

        sample_size = min(self.K, candidate_ind.size)
        k_choices = np.random.choice(
            np.arange(candidate_ind.size),
            sample_size, # This previously was hardcoded to 10 but I believe it should depend on K
            replace=False,
            p=p / p.sum(),
        )
        k_choice = k_choices[np.argmax(vals[k_choices])]

        acq_vals = np.zeros_like(candidate_ind, dtype=float)
        acq_vals[k_choice] = 1.0
        return acq_vals


class random(acquisition_function):
    '''
    Random choices
    '''
    def compute(self, u, candidate_ind):
        return np.random.rand(candidate_ind.size)