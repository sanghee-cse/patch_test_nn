import torch 
import torch.nn as nn
import numpy as np

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)

activation_dict = {
    'tanh': nn.Tanh,    
    'relu': nn.ReLU,
    'Gaussian': Gaussian,
}

class MLP(nn.Module):
    def __init__(self, num_hidden_layers=3, hidden_dim=64, activation='tanh', L=1.0):
        super(MLP, self).__init__()
        
        self.L = L
        activation_fn = activation_dict.get(activation, nn.Tanh)
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation_fn())

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inputs = torch.cat([x/self.L, y/self.L], dim=1)
        out = self.net(inputs)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5] 
