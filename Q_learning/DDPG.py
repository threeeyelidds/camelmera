import torch
import torch.nn as nn
import torch.nn.functional as F

# To tune network structure, simply add or delete the hidden_sizes list. 
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], activation_function=F.relu):
        super(Actor, self).__init__()
        self.activation_function = activation_function
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], activation_function=F.relu):
        super(Critic, self).__init__()
        self.activation_function = activation_function
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size + action_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return self.output_layer(x)


'''
Tune Hyperparameters
Increase the depth of the network by adding more hidden layers.
Increase the width of the network by using more units in each hidden layer.
Experiment with different activation functions, such as ReLU, Leaky ReLU, or Tanh.
Use techniques such as dropout or batch normalization to improve the network's ability to generalize.
'''