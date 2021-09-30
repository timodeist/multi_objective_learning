"""
This is code taken (with some changes) from the code base accompanying
Mahapatra, Debabrata, and Vaibhav Rajan.
"Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization."
International Conference on Machine Learning. PMLR, 2020.

LICENSE:
MIT License

Copyright (c) 2020 Debabrata Mahapatra

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
import cvxpy as cp
import cvxopt
import warnings

from .utils import *


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a


class EPO_LP(object):
    """
    class for EPO optimization of single solution
    """
    def __init__(self, m, n, r, eps=1e-4):
        """
        m = number of objectives
        n = number of parameters in the solution
        r = preference vector
        """       
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"

        return self.alpha.value


class EPO(object):
    """
    Main EPO optimizer class for multiple solutions
    """
    def __init__(self, n_mo_sol, n_mo_obj, n_parameters_list, preferences=None, eps=1e-4):
        """
        Inputs:
        n_mo_sol = number of solutions (networks)
        n_mo_obj = number of objectives
        n_parameters_list = number of paremeters in each solution
        preferences = preference vector for each solution (numpy array), n_mo_sol * n_mo_obj
                     or None
        """
        self.name = 'epo'
        if not (n_mo_obj == 2):
            warnings.warn('EPO has only been tested (probably works) for 2 objectives.')

        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.n_parameters_list = n_parameters_list
        if preferences is None:
            self.preferences = generate_k_preferences(n_mo_sol, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2)
        else:
            self.preferences = np.array(preferences, dtype=np.float32)
            assert self.preferences.shape==(n_mo_sol, n_mo_obj)
        print("preferences: ", self.preferences)

        self.epo_optimizer_list = list()
        for i_mo_sol in range(0, self.n_mo_sol):
            epo_lp = EPO_LP(self.n_mo_obj, self.n_parameters_list[i_mo_sol], self.preferences[i_mo_sol, :], eps=eps)
            self.epo_optimizer_list.append(epo_lp)


    def compute_weights(self, net_list, meta_optimizer_list, obj_func, input_data_batch, label_batch):
        n_mo_sol = self.n_mo_sol
        n_mo_obj= self.n_mo_obj
            
        weights = torch.zeros(n_mo_obj,n_mo_sol)
        for i_mo_sol in range(0,n_mo_sol):

            # compute gradients
            grads, losses = compute_grads_and_losses(
                        n_mo_obj,
                        net_list[i_mo_sol],
                        meta_optimizer_list[i_mo_sol],
                        obj_func,
                        input_data_batch,
                        label_batch)

            # run EPO
            weights[:, i_mo_sol] = self.compute_single_epo_weight_set(
                losses,
                grads,
                self.preferences[i_mo_sol,:],
                self.epo_optimizer_list[i_mo_sol])

        return weights


    def compute_single_epo_weight_set(self, losses, grads, preference, epo_lp):
            G = grads
            GG = G @ G.T
            # losses need to be a numpy array of size (2,0) 
            # losses = np.stack(losses.cpu())
            # losses = torch.stack(losses).cpu().detach().numpy()
            losses = losses.cpu().detach().numpy()
            # Instantiate EPO Linear Program Solver
            n_tasks = len(losses)

            try:
                # Calculate the alphas from the LP solver
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                # if epo_lp.last_move == "dom": # unused
                    # descent += 1 # unused
            except Exception as e:
                print(e)
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = preference / preference.sum()
                # n_manual_adjusts += 1 # unused

            alpha = n_tasks * torch.from_numpy(alpha).float()
            
            weights = alpha
            return(weights)