import os
import torch
import numpy as np
from tqdm import tqdm

from problems import mo_regression
from mo_optimizers import linear_scalarization, hv_maximization, pareto_mtl, epo
from mo_optimizers.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv


def validate_config(mo_optimizer_name, cfg):
    # conflict b/w mo_optimizer_name and mo_mode
    if mo_optimizer_name in ['pareto_mtl', 'epo'] and \
        cfg["mo_mode"] == "per_sample":
        raise ValueError(f"invalid combination in config. \
            mo_mode = 'per_sample' does not make sense for \
            mo_optimizer_name = {mo_optimizer_name}")
    elif mo_optimizer_name == 'linear_scalarization' and \
        cfg["mo_mode"] == "per_sample":
        raise NotImplementedError(f"invalid combination in config. \
            mo_mode = 'per_sample' not implemented for \
            mo_optimizer_name = {mo_optimizer_name}")


def sanity_check(val, min_val=0, max_val=np.infty):
    assert (not np.any(np.isnan(val)))
    assert np.all(val >= min_val)
    assert np.all(val < max_val)    
    return None


def initialize_net_and_optims(target_device, cfg):
    n_mo_sol = cfg["n_mo_sol"]

    net_list = []
    net_optimizer_list = []
    for i_mo_sol in range(0, n_mo_sol):
        net_list.append(mo_regression.Net())
        net_list[i_mo_sol].to(target_device)
        
        net_optimizer = torch.optim.Adam(net_list[i_mo_sol].parameters(), lr=cfg["lr"], betas=cfg["betas"])
        net_optimizer_list.append(net_optimizer)
    return net_list, net_optimizer_list


def initialize_mo_optimizer(mo_optimizer_name, target_device, cfg, net_list):
    """
    calls init method of respective mo_optimizer class with appropriate args
    """
    ref_point = cfg["ref_point"]
    n_mo_sol = cfg["n_mo_sol"]
    n_mo_obj = cfg["n_mo_obj"]

    if mo_optimizer_name == "linear_scalarization":
        mo_opt = linear_scalarization.LinearScalarization(n_mo_sol, n_mo_obj)
    elif mo_optimizer_name == "hv_maximization":
        mo_opt = hv_maximization.HvMaximization(n_mo_sol, n_mo_obj, ref_point)
    elif mo_optimizer_name == "pareto_mtl":
        mo_opt = pareto_mtl.ParetoMTL(n_mo_sol, n_mo_obj, device=target_device)
    elif mo_optimizer_name == "epo":
        n_parameters_list = list()
        for i_mo_sol in range(n_mo_sol):
            n_parameters_list.append(int(np.sum([cur_par.numel() for cur_par in net_list[i_mo_sol].parameters()])))
        mo_opt = epo.EPO(n_mo_sol, n_mo_obj, n_parameters_list)
    else:
        raise ValueError("unknown opt name")
    return mo_opt


def forward_propagation(net_list, net_optimizer_list, loss_fn, inputs, targets):
    """
    compute loss for single forward propagation
    """

    loss_torch = []  #sol, obj
    loss_numpy = []
    loss_torch_per_sample = [] #sol, obj, n_samples
    loss_numpy_per_sample = []
    for i_mo_sol in range(len(net_list)):
        net_optimizer_list[i_mo_sol].zero_grad()
        Y_hat = net_list[i_mo_sol](inputs)
        loss_per_sample = loss_fn(Y_hat, targets)  #obj, n_samples

        loss_torch_per_sample.append(loss_per_sample)
        loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())

        loss_mean = torch.mean(loss_per_sample, dim=1)
        loss_torch.append(loss_mean)
        loss_numpy.append(loss_mean.cpu().detach().numpy())

    # change axis order in numpy arrays for consistency later
    loss_numpy = np.array(loss_numpy).T  #obj, sol
    loss_numpy_per_sample = np.array(loss_numpy_per_sample).transpose(2, 1, 0) #n_samples, obj, sol
    
    # check validity of loss values
    sanity_check(loss_numpy)
    return loss_torch, loss_numpy, loss_torch_per_sample, loss_numpy_per_sample


def dynamic_weight_optimization_per_sample(mo_opt, \
                                        net_list, net_optimizer_list, \
                                        loss_fn,\
                                        train_x, train_y,\
                                        target_device):
    # forward propagation
    outs = forward_propagation(net_list, net_optimizer_list, loss_fn,\
                                train_x, train_y)
    _, _, loss_torch_per_sample, loss_numpy_per_sample = outs
    n_samples, n_mo_obj, n_mo_sol = loss_numpy_per_sample.shape

    # compute dynamic weights per sample
    dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
    for i_sample in range(0, n_samples):
        weights = mo_opt.compute_weights(loss_numpy_per_sample[i_sample,:,:])
        dynamic_weights_per_sample[:, :, i_sample] = weights.permute(1,0)

    dynamic_weights_per_sample = dynamic_weights_per_sample.to(target_device)

    # backward propagation
    for i_mo_sol in range(0, len(net_list)):
        dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :] * loss_torch_per_sample[i_mo_sol], dim=0))
        dynamic_loss.backward()
        net_optimizer_list[i_mo_sol].step()

    return outs


def dynamic_weight_optimization_average(mo_opt, \
                                        net_list, net_optimizer_list, \
                                        loss_fn,\
                                        train_x, train_y,\
                                        target_device):
    # dynamic weight calculation in epo and pareto MTL requires gradients, \
    # so needs extra forward & backward propagation
    if mo_opt.name in ['pareto_mtl', 'epo']:
        dynamic_weights = mo_opt.compute_weights(net_list, net_optimizer_list, loss_fn, \
                                                    train_x, train_y)

    # forward propagation
    outs = forward_propagation(net_list, net_optimizer_list, loss_fn,\
                                train_x, train_y)
    loss_torch, loss_numpy, _, _ = outs

    # compute dynamic weights
    if mo_opt.name in ['hv_maximization', 'linear_scalarization']:
        dynamic_weights = mo_opt.compute_weights(loss_numpy)

    dynamic_weights = dynamic_weights.to(target_device)
    # backward propagation
    for i_mo_sol in range(0, len(net_list)):
        dynamic_loss = torch.sum(dynamic_weights[:, i_mo_sol] * loss_torch[i_mo_sol])
        dynamic_loss.backward()
        net_optimizer_list[i_mo_sol].step()
    
    return outs


def run(mo_optimizer_name, target_device, cfg):
    validate_config(mo_optimizer_name, cfg)
    ref_point = cfg["ref_point"]
    n_mo_sol = cfg["n_mo_sol"]
    n_mo_obj = cfg["n_mo_obj"]
    n_learning_iterations = cfg["n_learning_iterations"]    

    # ---- initialize net_list, data, loss etc ----
    net_list, net_optimizer_list = initialize_net_and_optims(target_device, cfg)
    train_x, train_y, validation_x, validation_y = mo_regression.load_datasets(target_device, cfg)
    loss_fn = mo_regression.initialize_losses(cfg)
    mo_opt = initialize_mo_optimizer(mo_optimizer_name, target_device, cfg, net_list)

    # ---- training ----
    training_loss = []
    training_hv = []
    for niter in tqdm(range(n_learning_iterations), desc="iter"):
        if cfg["mo_mode"]=="per_sample":
            outs = dynamic_weight_optimization_per_sample(mo_opt, \
                                                        net_list, net_optimizer_list, \
                                                        loss_fn,\
                                                        train_x, train_y,\
                                                        target_device)
            
            _, loss_numpy, _, loss_numpy_per_sample = outs
            training_loss.append(loss_numpy)
            hv = 0.
            for i_sample in range(loss_numpy_per_sample.shape[0]):
                hv += compute_hv(loss_numpy_per_sample[i_sample, :, :], ref_point)
            training_hv.append(hv/float(loss_numpy_per_sample.shape[0]))
        
        elif cfg["mo_mode"]=="average":
            outs = dynamic_weight_optimization_average(mo_opt, \
                                                        net_list, net_optimizer_list, \
                                                        loss_fn,\
                                                        train_x, train_y,\
                                                        target_device)            
            
            _, loss_numpy, _, loss_numpy_per_sample = outs
            training_loss.append(loss_numpy)
            training_hv.append(compute_hv(loss_numpy, ref_point))

    training_loss = np.array(training_loss)  #n_learning_iterations * n_mo_obj * n_mo_sol
    training_hv = np.array(training_hv)  #n_learning_iterations

    # ---- validation ----
    for net in net_list:
        net.eval()

    validation_output = []
    validation_loss = []
    for i_mo_sol in range(len(net_list)):
        Y_hat = net_list[i_mo_sol](validation_x)
        loss = loss_fn(Y_hat, validation_y) #n_mo_obj * n_samples
        
        validation_output.append(Y_hat.view(-1).cpu().detach().numpy())
        validation_loss.append(loss.cpu().detach().numpy())

    validation_output = np.array(validation_output) #n_mo_sol * n_samples
    validation_loss = np.array(validation_loss).transpose(2,1,0)   #n_samples * n_mo_obj * n_mo_sol
    validation_loss_mean = validation_loss.mean(axis=0)
    validation_hv = compute_hv(validation_loss_mean, ref_point)

    output_dict = {"mo_optimizer_name": mo_optimizer_name,
                    "training_loss": training_loss,
                    "training_hv": training_hv,
                    "validation_data_x": validation_x.view(-1).cpu().numpy(),
                    "validation_data_y": validation_y.view(-1, n_mo_obj).cpu().numpy(),
                    "validation_output": validation_output,
                    "validation_loss": validation_loss,
                    "validation_hv": validation_hv
                    }
    return output_dict


if __name__ == '__main__':
    pass



