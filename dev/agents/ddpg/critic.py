"""
    Critic network for DDPG algorithm.

    @author: j-huthmacher
"""


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Critic(nn.Module):
    """
        This class represents the neural network that is used as the critic for
        the DDPG agent.
    """

    def __init__(self, input_dim: int, hidden_dim: [int], output_dim: int):
        """ Initialize critic.

            Paramters:
            ----------
                input_dim: int
                    Input dimension of the input layer.
                hidden_dim: [int]
                    Array of input dimensions for the hidden layers.
                ouput_dim: int
                    Output dimension of the network.

        """

        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])

        self.hidden_layer = nn.ModuleList()

        last_dim = hidden_dim[0]
        for dim in hidden_dim[1:]:
            self.hidden_layer.append(nn.Linear(last_dim, dim))
            last_dim = dim

        self.output_layer = nn.Linear(last_dim, output_dim)

    def init_weight(self):
        """ TODO
        """

    def forward(self, state: np.array, action: np.array):
        """ Forward pass.

            Parameters:
            -----------
                state: numpy.array
                    Current state.
                action: numpy.array
                    Action taken in the current state.
            Return:
            -------
                numpy.array: Feedback for taking the specific action in that
                specific state.
        """

        # x = torch.cat([state, action.squeeze(1)], 1)
        x = torch.cat([state, action], 1)
        x = F.relu(self.input_layer(x))

        for layer in self.hidden_layer:
            x = F.relu(layer(x))

        x = torch.tanh(self.output_layer(x))

        return x
