"""
This code is in most parts taken (and then adapted) from the code base accompanying the manuscript:
Lin, Xi, et al.
"Pareto multi-task learning."
arXiv preprint arXiv:1912.12854 (2019).

LICENSE:
MIT License

Copyright (c) 2019 Xi Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import torch
from mo_optimizers.min_norm_solvers import MinNormSolver
from .utils import *


class ParetoMTL(object):
    """
    Main class for Pareto MTL optimizer. only works for 2 objectives
    """
    def __init__(self, n_mo_sol, n_mo_obj, preferences=None, device="cuda:0"):
        """
        Inputs:
        n_mo_sol = number of solutions (networks)
        n_mo_obj = number of objectives
        preferences = preference vector for each solution (numpy array), n_mo_sol * n_mo_obj
                     or None
        """
        self.name = 'pareto_mtl'
        self.device = device
        if not (n_mo_obj == 2):
            raise NotImplementedError('ParetoMTL works for 2 objectives.')

        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        if preferences is None:
            self.preferences = generate_k_preferences(n_mo_sol, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2)
            self.preferences = torch.tensor(self.preferences, dtype=torch.float32, device=device)
        else:
            self.preferences = torch.tensor(preferences, dtype=torch.float32, device=device)
            assert self.preferences.shape==(n_mo_sol, n_mo_obj)
        print("preferences: ", self.preferences)

        self.flag_list = [False for i in range(n_mo_sol)]
        self.iter = 0


    def compute_weights(self, net_list, meta_optimizer_list, obj_func, input_data_batch, label_batch):
        self.iter += 1
        n_mo_sol = self.n_mo_sol
        n_mo_obj= self.n_mo_obj

        weights = torch.zeros(n_mo_obj, n_mo_sol)
        for i_mo_sol in range(0, n_mo_sol):

            # compute gradients
            grads, losses = compute_grads_and_losses(
                        n_mo_obj,
                        net_list[i_mo_sol],
                        meta_optimizer_list[i_mo_sol],
                        obj_func,
                        input_data_batch,
                        label_batch)

            if (not self.flag_list[i_mo_sol]) and (self.iter <= 1000):
                # run init search Pareto MTL
                self.flag_list[i_mo_sol], weights[:, i_mo_sol] = self.run_pareto_mtl_init(
                                                                                            grads,
                                                                                            losses,
                                                                                            self.preferences,
                                                                                            i_mo_sol)

            # if flag is True OR turned True in this iteration, run Pareto MTL
            if self.flag_list[i_mo_sol] or self.iter > 1000:
                weights[:, i_mo_sol] = self.run_pareto_mtl(
                                                            grads,
                                                            losses,
                                                            self.preferences,
                                                            i_mo_sol)

        return weights

    def run_pareto_mtl_init(self, grads, losses_vec, ref_vec, pref_idx):
        """ 
        Find weights to optimize for feasible initialization
        """
        # calculate the weights
        flag, weight_vec = self.get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
        
        # early stop once a feasible solution is obtained
        if flag == True:
            print("feasible solution is obtained.")
        
        return(flag,weight_vec)

    def run_pareto_mtl(self, grads, losses_vec, ref_vec, pref_idx):
        n_tasks = len(losses_vec)
        # calculate the weights
        weight_vec = self.get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
        
        normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
        return(weight_vec)

        
    def get_d_paretomtl_init(self, grads, value, weights, i):
        """ 
        calculate the gradient direction for ParetoMTL initialization
        value is loss
        weights is preference
        """
        flag = False
        nobj = value.shape
       
        # check active constraints
        current_weight = weights[i]
        rest_weights = weights
        w = rest_weights - current_weight
        
        gx =  torch.matmul(w,value/torch.norm(value))
        idx = gx >  0
       
        # calculate the descent direction
        if torch.sum(idx) <= 0:
            flag = True
            return flag, torch.zeros(nobj)
        if torch.sum(idx) == 1:
            sol = torch.ones(1, dtype=torch.float32).to(self.device)
        else:
            vec =  torch.matmul(w[idx],grads)
            sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        # only works for 2 losses!
        weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
        weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
        weight = torch.stack([weight0,weight1])
        return flag, weight


    def get_d_paretomtl(self, grads, value, weights, i):
        """
        calculate the gradient direction for ParetoMTL
        value = losses
        weights = pref vector
        i = current

        """
        # check active constraints
        current_weight = weights[i]
        rest_weights = weights 
        w = rest_weights - current_weight
        
        gx =  torch.matmul(w,value/torch.norm(value))
        idx = gx >  0
        

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.tensor(sol).float().to(self.device)
        ## if not all networks are in their assigned subspace, also consider constraints to push networks back into their subspace

        # vec contains the loss gradients and the gradients of G, which used in the active feasibility constraints
        # (one constraint for each combination of network that is outside its subspace and the preference vector of the subspace to which it is closer than its assigned subspace)
        # e.g. if 1 network has a more actue angle to two preference vectors other than its assigned preference vector, then you have 2 additional active constraints
        vec =  torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        # this only works for 2 losses!
        # my interpretation(!): when there are active constraints, these need to be taken into account when computing weights for the 2 loss functions used for backpropagation
        # \Nabla G in Eq (14) of Lin et al (2019) is composed of a weighted combination of  \Nabla L_0 and \Nabla L_1. All the weights corresponding to \Nabla L_i need to be accumulated to create the loss weights.
        # Example: w[idx][0,a] with a = 0 are the weights of L_a of all active constraints. Say, G_0 is active, then \Nabla G[0] is =  w[0][0,0] \Nabla L_0 +  w[0][0,1] \Nabla L_1.
        # If also G_1 is active then, \Nabla G[1] is =  w[1][0,0] \Nabla L_0 +  w[1][0,1] \Nabla L_1.
        # All these needs to be collected for L_0 and L_1
        # I have not yet figured out yet how the idx business works precisely.
        weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight = torch.stack([weight0,weight1])
        
        return weight
