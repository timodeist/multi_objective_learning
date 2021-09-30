import numpy as np
import torch
from .utils import *


class LinearScalarization(object):
    """
    Linear Scalarization MO optimizer. Currently works for two objectives only.
    """
    def __init__(self, n_mo_sol, n_obj, weights=None):
        self.name = 'linear_scalarization'
        if weights is None:
            # generate_fixed_weights(n_obj, n_mo_sol)  #this is the method we use
            weights = generate_k_preferences(n_mo_sol, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2) #epo like method
            weights = weights.T
            weights = torch.from_numpy(weights.copy()).float()
        else:
            weights = torch.tensor(weights).float()
        
        assert weights.shape==(n_obj, n_mo_sol)
        self.weights = weights
        print("fixed weights: ", self.weights)


    def compute_weights(self, mo_obj_val):
        return self.weights