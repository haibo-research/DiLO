import torch
import numpy as np
import torch.nn as nn
import operator
from functools import reduce
from matplotlib import pyplot as plt
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reading data
class NpyReader(object):
    def __init__(self, file_path, to_cuda=False):
        super(NpyReader, self).__init__()

        self.file_path = file_path
        self.data = np.load(file_path, allow_pickle=True).item()
        self.keys = list(self.data.keys())
        self.device = torch.device('cuda') if to_cuda else torch.device('cpu')
        
    def read_field(self, key):
        if key not in self.keys:
            raise KeyError(f"Field '{key}' not found in data. Available keys: {self.keys}")
        
        field_data = self.data[key]
        
        # Ensure data is float type and convert to Tensor
        if not isinstance(field_data, torch.Tensor):
            field_data = torch.from_numpy(field_data).float()

        return field_data.to(self.device)

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x=None, mean=None, std=None, eps=1e-5, device=None):
        if x is not None:
            # When a data tensor is provided, compute mean and std from it.
            # This is typically done during training.
            self.mean = torch.mean(x, [0, 1, 2]).squeeze()
            self.std = torch.std(x, [0, 1, 2]).squeeze()
        elif mean is not None and std is not None:
            # When mean and std are provided directly, use them.
            # This is typically done during inference/loading.
            self.mean = mean
            self.std = std
        else:
            raise ValueError("Either 'x' (a data tensor) or both 'mean' and 'std' must be provided.")
            
        self.eps = eps
        
        if device:
            self.to(device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        # The sample_idx logic is primarily for UnitGaussianNormalizer,
        # but we keep a simplified version for consistency.
        # For this global normalizer, we ignore sample_idx.
        return (x * (self.std + self.eps)) + self.mean

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x



# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


if __name__ == '__main__':
    ntest=2
    TRAIN_PATH = '/home/hbliu/dl2ip_code/FNO/data/train_data.npy'
    
    # Test the data loading
    if os.path.exists(TRAIN_PATH):
        print(f"Testing data loading from {TRAIN_PATH}")
        reader = NpyReader(TRAIN_PATH)
        try:
            sigma = reader.read_field('sigma')
            u = reader.read_field('u')
            print(f"Successfully loaded sigma with shape {sigma.shape}")
            print(f"Successfully loaded u with shape {u.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print(f"Test file {TRAIN_PATH} doesn't exist")