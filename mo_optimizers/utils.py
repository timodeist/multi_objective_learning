"""
    Pareto MTL LICENSE:
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

"""
    EPO LICENSE:
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
import torch
from torch.autograd import Variable


def compute_grads_and_losses(n_mo_obj,cur_net,optimizer,obj_func,input_data_batch,label_batch):
    """
    computes gradients for each objective
    Adapted from Pareto MTL and EPO codebases, see licenses at top of file
    """
    grads = dict()
    for i_mo_obj in range(0,n_mo_obj):
        optimizer.zero_grad()
        Y_hat = cur_net(input_data_batch)
        loss_per_sample = obj_func(Y_hat, label_batch)
        losses = loss_per_sample.mean(dim=1)
        losses[i_mo_obj].backward()

        grads[i_mo_obj] = []
        if hasattr(cur_net,"style_layers"):
            param = cur_net.params
            if param.grad is not None:
                grads[i_mo_obj].append(param.grad.data.clone().flatten())
        else:
            for param in cur_net.parameters():
                if param.grad is not None:
                    grads[i_mo_obj].append(param.grad.data.clone().flatten())

    grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
    grads = torch.stack(grads_list)

    # check that losses are not nan and [0,infty)
    for entry in losses:
        assert (not torch.any(torch.isnan(entry)))
        assert torch.all(entry >= 0)
        assert torch.all(torch.isfinite(entry))

    # check that losses are not nan and (-infty,infty)
    assert (not torch.any(torch.isnan(grads)))
    assert torch.all(torch.isfinite(grads))

    return(grads, losses)


def generate_k_preferences(K, min_angle=None, max_angle=None):
    """
    generate evenly distributed preference vector
    Adapted from EPO codebase, see license at top of file
    """
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y].astype(np.float32)


def generate_fixed_weights(n_obj, n_mo_sol):
    """
    inputs:
    n_obj, n_mo_sol

    outputs:
    weights: n_obj * n_mo_sol, tensor
    """
    if n_obj==1:
        weights = torch.ones(n_mo_sol).view(1, -1)
    elif n_obj==2:
        if n_mo_sol==1:
            weights = torch.ones(n_obj, n_mo_sol)
        else:
            weights = torch.zeros(n_obj, n_mo_sol)
            for i_mo_sol in range(0, n_mo_sol):
                weights[0, i_mo_sol] = i_mo_sol/(n_mo_sol-1)
                weights[1, i_mo_sol] = 1 - weights[0, i_mo_sol]
    else:
        raise ValueError('generating fixed weights is not yet generalized to more than 2 objectives')

    return weights