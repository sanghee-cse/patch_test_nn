import torch 
import torch.nn as nn
import numpy as np

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)
    

class Sine(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = 5.0 

    def forward(self, x):
        return torch.sin(self.omega * x)
        
activation_dict = {
    'tanh': nn.Tanh,    
    'relu': nn.ReLU,
    'Gaussian': Gaussian,
    'sin': Sine,
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

class SkewMLP(nn.Module):
    # This contaminates the y-coordinate inside this model
    def __init__(self, num_hidden_layers=3, hidden_dim=64, activation='tanh', L=1.0):
        super(SkewMLP, self).__init__()
        
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
        # x' = x
        # y' = y + 0.2 * x 
        
        x_distorted = x / self.L
        y_distorted = (y / self.L) + 0.2 * (x.detach() / self.L)
        
        inputs = torch.cat([x_distorted, y_distorted], dim=1)
        
        out = self.net(inputs)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
    
class SxxMLP(nn.Module):
    def __init__(self, num_hidden_layers=3, hidden_dim=64, activation='tanh', L=1.0):
        super().__init__()

        self.L = L
        activation_fn = activation_dict.get(activation, nn.Tanh)
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation_fn())

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_dim, 4))
        self.main_net = nn.Sequential(*layers)

        self.sxx_net = nn.Sequential(
            nn.Linear(2, 32), 
            activation_fn(),
            nn.Linear(32, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x/self.L, y/self.L], dim=1)
        out  = self.main_net(inputs)
        sxx = self.sxx_net(torch.cat([x/self.L, y/self.L], dim=1)) 
        
        return out[:, 0:1], out[:, 1:2], sxx, out[:, 2:3], out[:, 3:4]
        
class SyyMLP(nn.Module):
    def __init__(self, num_hidden_layers=3, hidden_dim=64, activation='tanh', L=1.0):
        super().__init__()

        self.L = L
        activation_fn = activation_dict.get(activation, nn.Tanh)
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation_fn())

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_dim, 4))
        self.main_net = nn.Sequential(*layers)

        self.syy_net = nn.Sequential(
            nn.Linear(2, 32), 
            activation_fn(),
            nn.Linear(32, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x/self.L, y/self.L], dim=1)
        out  = self.main_net(inputs)
        syy = self.syy_net(torch.cat([x/self.L, y/self.L], dim=1)) 
        
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], syy, out[:, 3:4]     