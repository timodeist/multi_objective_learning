import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_trigonometric_dataset(n_samples=400, n_targets=2, cycles=1):
    X = np.random.uniform(0, cycles * 2 * np.pi, n_samples)
    Y = np.zeros((n_samples, n_targets))

    Y[:, 0] = np.cos(X)
    Y[:, 1] = np.sin(X)
    if n_targets==3:
        Y[:, 2] = np.sin(X + 1*np.pi)

    # needs extra dimension so that a NN recognizes these as a batch
    X = X[:, None]
    return X, Y


def train_and_val_split(data_x, data_y, train_ratio=0.5):
    nsamples = len(data_x)
    print("total data: ", nsamples)
    indices = np.arange(nsamples)
    np.random.shuffle(indices)

    train_indices = indices[:int(nsamples*train_ratio)]
    val_indices = indices[int(nsamples*train_ratio):]

    train_x, train_y = data_x[train_indices], data_y[train_indices]
    validation_x, validation_y = data_x[val_indices], data_y[val_indices]
    print("training data: {}, validation data: {}".format(
            len(train_x), len(validation_x)))
    return train_x, train_y, validation_x, validation_y


def load_datasets(target_device, cfg):
    n_samples = cfg["n_samples"]
    n_targets = cfg["n_mo_obj"]
    train_ratio = cfg["train_ratio"]
    data_x, data_y = generate_trigonometric_dataset(n_samples=n_samples, n_targets=n_targets)
    train_x, train_y, validation_x, validation_y = train_and_val_split(data_x, data_y, train_ratio=train_ratio)

    train_x = torch.from_numpy(train_x).float().to(target_device)
    train_y = torch.from_numpy(train_y).float().to(target_device)
    validation_x = torch.from_numpy(validation_x).float().to(target_device)
    validation_y = torch.from_numpy(validation_y).float().to(target_device)
    return train_x, train_y, validation_x, validation_y


class Net(nn.Module):
    def __init__(self, n_intermediate_layers=2, n_neurons=100):
        super().__init__()
        n_features_in = 1
        n_outputs = 1
        
        cur_features = n_features_in
        layer_list = list()
        for _ in range(0, n_intermediate_layers):
                layer_list.append(torch.nn.Linear(cur_features, n_neurons))
                cur_features = n_neurons

        self.layer_list = nn.ModuleList(layer_list)
        self.linear_out = torch.nn.Linear(cur_features, n_outputs)

    def forward(self, inputs):
        outs = inputs
        for layer in self.layer_list:
            outs = layer(outs)
            outs = F.relu(outs)

        model_out = self.linear_out(outs)
        return(model_out)


class ScaledMSELoss(nn.Module):
    """mse loss scaled by 0.01"""
    def __init__(self, reduction='none'):
        super(ScaledMSELoss, self).__init__()
        self.reduction = reduction


    def forward(self, inputs, target):
        """
        out = 0.01 * mse_loss(inputs, target)
        """
        out = 0.01 * torch.nn.functional.mse_loss(inputs, target, reduction=self.reduction) 
        return out


class Loss(nn.Module):
    """Evaluation of two losses"""
    def __init__(self, loss_name_list):
        super(Loss, self).__init__()
        self.implemented_loss = ["MSELoss", "L1Loss", "ScaledMSELoss"]

        self.loss_list = []
        for loss_name in loss_name_list:
            if loss_name not in self.implemented_loss:
                raise NotImplementedError("{} not implemented. Implemented losses are: {}".format(loss_name, self.implemented_loss))
            elif loss_name == "MSELoss":
                self.loss_list.append( torch.nn.MSELoss(reduction='none') )
            elif loss_name == "L1Loss":
                self.loss_list.append( torch.nn.L1Loss(reduction='none') )
            elif loss_name == "ScaledMSELoss":
                self.loss_list.append( ScaledMSELoss(reduction='none') )

    def forward(self, inputs, target):
        """
        out_list = list of losses, where each loss is a tensor of losses for each sample
        """
        assert(target.shape[1] == len(self.loss_list))
        target = target.to(inputs.device)
        out_list = []
        for i, loss_fn in enumerate(self.loss_list):
            out = loss_fn(inputs, target[:, i][:, None])
            out_list.append(out.view(-1))

        out = torch.stack(out_list, dim=0)
        return out


def initialize_losses(cfg):
    loss_names = cfg["loss_names"]
    loss_fn = Loss(loss_names)
    return loss_fn


if __name__ == '__main__':
    pass

