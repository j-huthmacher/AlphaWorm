"""
    Actor network for DDPG algorithm.

    @author: j-huthmacher
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from config.config import log


class Actor(nn.Module):
    """
        This class represents the neural network that is used as the actor for
        the DDPG agent.
    """

    def __init__(self, input_dim: int, hidden_dim: [int], output_dim: int,
                 max_action: float = 1.):
        """ Initialize actor.

            Paramters:
            ----------
                input_dim: int
                    Input dimension of the input layer.
                hidden_dim: [int]
                    Array of input dimensions for the hidden layers.
                ouput_dim: int
                    Output dimension of the network.
                max_action: [int]
                    Containing the maximum action for each dimension in the
                    action vector.

        """

        super(Actor, self).__init__()

        self.max_action = max_action  # torch.from_numpy(np.asarray(max_action)).float()

        self.input_layer = nn.Linear(input_dim, hidden_dim[0])

        self.hidden_layer = nn.ModuleList()

        last_dim = hidden_dim[0]
        for dim in hidden_dim[1:]:
            self.hidden_layer.append(nn.Linear(last_dim, dim))
            last_dim = dim

        self.output_layer = nn.Linear(last_dim, output_dim)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)

    def forward(self, state: np.array):
        """ Forward pass.

            Parameters:
            -----------
                state: numpy.array
                    Current state for which we want to predict the next action.
            Return:
            -------
                numpy.array: Predicted action in form of a vector.
        """
        x = F.relu(self.input_layer(state))

        for layer in self.hidden_layer:
            x = F.relu(layer(x))

        # Action space between -1 and 1
        x = torch.tanh(self.output_layer(x))

        # Should not change anything since we are already between -1 and 1
        x = x * self.max_action

        return x
