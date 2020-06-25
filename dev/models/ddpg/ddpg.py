"""
    Implementation of the DDPG algorithmus.

    @paper: https://arxiv.org/pdf/1509.02971.pdf
    @author: j-huthmacher
"""

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import gym
import numpy as np

from dev.models import PrioritizingReplayBuffer
from dev.models import MemoryBuffer

from dev.models.ddpg.Actor import Actor
from dev.models.ddpg.Critic import Critic



class DDPGagent:
    """
        Implementation of an DDPG-Agent that uses the DDPG algorithm
        for learning.
    """
    def __init__(self, env: gym.Env, hidden_dim: [int] = [256],
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 1e-3,
                 max_memory: int = 5000):
        """ Initialization of the DDPG-Agent

            Parameters:
            -----------
                env: gym.Env
                    Environemnt where the agent is located in.
                hidden_dim: [int]
                    Array of hidden (input) dimensions for the actor- and
                    critic-network.
                actor_lr: float
                    Learning rate for the actor network.
                critic_lr: float
                    Learning rate for the critic network.
                gamma: float
                    Discount factor.
                tau: float
                    Factor for the soft target updates.
                max_memory: float
                    Maximal size of the memory buffer (replay buffer).
        """

        self.num_actions = env.action_space.shape[0]
        #self.num_states = env.observation_space.shape[0]

        #self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]


        self.gamma = gamma
        self.tau = tau

        # Initialize actor- and critic networks
        self.actor = Actor(self.num_states, hidden_dim, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_dim,
                                  self.num_actions)
        # Make copy of network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # In the critic network we consider also the action to give "feedback"
        # Therefore the input dimension of self.num_states + self.num_actions
        self.critic = Critic(self.num_states + self.num_actions,
                             hidden_dim, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions,
                                    hidden_dim, self.num_actions)
        # Make copy of network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # initialize training set up

        # For normal random sampling
        #self.memory_buffer = MemoryBuffer.MemoryBuffer(max_memory)

        # For prioritized experience replay

        self.memory_buffer = PrioritizingReplayBuffer.MemoryBuffer(max_memory)


        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

    def get_action(self, state: np.array):
        """ Returns the action that should be taken in the current state.
            Parameters:
            -----------
                state: np.array
                    The current state which is used to choose the next actions.
            Return:
            -------
                np.array: The action in form of vector predicted by the actor
                          component.
        """
        s = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(s).detach().numpy()
        return action

    def update(self, batch_size: int = 64, tau: float = None):
        """ Function to update the actor and critic components.

            Parameters:
            -----------
                batch_size: int
                    The number of samples we want to sample from the memory
                    buffer (replay buffer). For PRB = 128
        """

        if len(self.memory_buffer.buffer) < batch_size:
            return

        # state, action, reward, next_state
        s, a, r, next_s, done, importance, indices = self.memory_buffer.sample(batch_size)

        # Create CPU tensors (for GPU use torch.cuda.FloatTensor)
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        next_s = torch.FloatTensor(next_s)


        # Calculate the critic loss ("feedback")
        # Old q value -> next action -> next q value -> loss
        q_values = self.critic.forward(s, a)
        next_actions = self.actor_target.forward(next_s).detach()
        next_q = self.critic_target.forward(next_s, next_actions.unsqueeze(2))
        q_prime = r + self.gamma * next_q
        errors = abs(q_values - q_prime)
        self.memory_buffer.set_priorities(indices, errors)
        critic_loss = self.critic_loss_func(q_values, q_prime)
        #critic_loss = self.critic_loss_func(q_values, q_prime)*importance

      #  TODO: Use critic_loss aka mean squared error for calculating the sample_probs


        # Calculate the actor loss
        actor_loss = -self.critic.forward(s, self.actor.forward(s).unsqueeze(2)).mean()

        # Updates!
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()        

        if not tau:
            tau = self.tau

        # Updates for the target networsk
        for t_param, param in zip(self.actor_target.parameters(),
                                  self.actor.parameters()):
            update = tau * param.data + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)

        for t_param, param in zip(self.critic_target.parameters(),
                                  self.critic.parameters()):
            update = tau * param.data + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)
