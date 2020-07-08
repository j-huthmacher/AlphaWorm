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

from agents.agent import Agent
from agents.memory_buffer import MemoryBuffer
from agents.ddpg.actor import Actor
from agents.ddpg.critic import Critic
from agents.ddpg.ou_noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGagent(Agent):
    """
        Implementation of an DDPG-Agent that uses the DDPG algorithm
        for learning.
    """
    def __init__(self, env: gym.Env, hidden_dim: [int] = [256],
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 1e-3,
                 max_memory: int = int(1e6), w_decay: float = 0.01,
                 normalize_obs: bool = True):
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
        self.max_action = float(env.action_space.high[0]) #env.action_space.high
        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.policy_noise = 0.2
        self.noise_clip = 0.5

        # Initialize actor- and critic networks
        self.actor = Actor(self.num_states, hidden_dim, self.num_actions,
                           self.max_action).to(device)
        self.actor_target = Actor(self.num_states, hidden_dim,
                                  self.num_actions, self.max_action).to(device)
        # Make copy of network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # In the critic network we consider also the action to give "feedback"
        # Therefore the input dimension of self.num_states + self.num_actions
        self.critic = Critic(self.num_states + self.num_actions,
                             hidden_dim, self.num_actions).to(device)
        self.critic_target = Critic(self.num_states + self.num_actions,
                                    hidden_dim, self.num_actions).to(device)
        # Make copy of network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # initialize training set up
        self.memory_buffer = MemoryBuffer(max_memory)

        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr,
                                          weight_decay=w_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr, weight_decay=w_decay)

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
        # s = Variable(torch.from_numpy(state).float().unsqueeze(0))
        s = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # self.actor(state).cpu().data.numpy().flatten()

        action = self.actor(s).cpu().data.numpy().flatten()
        return action

    def update(self, batch_size: int = 64, tau: float = None):
        """ Function to update the actor and critic components.

            Parameters:
            -----------
                batch_size: int
                    The number of samples we want to sample from the memory
                    buffer (replay buffer). For PRB = 128
        """

        if len(self.memory_buffer) < batch_size:
            return

        # state, action, reward, next_state
        # s, a, r, next_s, _ = self.memory_buffer.sample(batch_size)
        s, a, next_s, r, not_done = self.memory_buffer.sample(batch_size)


        # Create CPU tensors (for GPU use torch.cuda.FloatTensor)
        # The new replay buffer implementation returns directly tensors
        # s = torch.FloatTensor(s)
        # a = torch.FloatTensor(a)
        # r = torch.FloatTensor(r)
        # next_s = torch.FloatTensor(next_s)

        noise = (
                    torch.randn_like(a) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (
                        self.actor_target(next_s) + noise
                      ).clamp(-self.max_action, self.max_action)

        # Calculate the critic loss ("feedback")
        # Old q value -> next action -> next q value -> loss
        q_values = self.critic(s, a)
        # next_action = self.actor_target(next_s).detach()
        next_q = self.critic_target(next_s, next_action)  # unsqueeze(2)
        q_prime = r + (self.gamma * next_q)
        critic_loss = self.critic_loss_func(q_values, q_prime)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate the actor loss
        actor_loss = -self.critic(s, self.actor(s)).mean()

        # Updates!
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if not tau:
            tau = self.tau

        # Updates for the target networsk
        for t_param, param in zip(self.actor_target.parameters(),
                                  self.actor.parameters()):
            update = (tau * param.data) + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)

        for t_param, param in zip(self.critic_target.parameters(),
                                  self.critic.parameters()):
            update = (tau * param.data) + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)

    def run(self, env: object, steps: int = 100, render: bool = True):
        """ Method to execute a trained agent on a domain.

            Parameters:
            -----------
                env: UnityEnvironment or GymEnvironment

                steps: int
                    Number of actions the agent should take in this run.
                render: bool
                    Flag to decide if the environment is rendered while
                    the agent is active.
        """
        noise = OUNoise(env.action_space)
        state = env.reset()
        noise.reset()

        for step in range(steps):
            if render:
                env.render()
            action = self.get_action(state)
            action = noise.get_action(action, step)
            state, reward, done, _ = env.step(action)


def normalize(x, stats):
    """
    """
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)
