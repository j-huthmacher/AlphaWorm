"""
    Implementation of the DDPG algorithmus.

    @paper: https://arxiv.org/pdf/1509.02971.pdf
    @author: j-huthmacher
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
# from torch import Variable

from agents.agent import Agent
from agents import Actor, Critic  # , MemoryBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGagent(Agent):
    """
        Implementation of an DDPG-Agent that uses the DDPG algorithm
        for learning.
    """
    def __init__(self, env: any, hidden_dim: [int] = [256],
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 1e-3,
                 max_memory: int = int(1e6), w_decay: float = 0.01,
                 normalize_obs: bool = True):
        """ Initialization of the DDPG-Agent

            Parameters:
            -----------
                env: gym.Env or Unity Environment
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
        self.max_action = float(env.action_space.high[0])
        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.policy_noise = 0.2
        self.noise_clip = 0.5

        ##################
        # Actor Networks #
        ##################
        self.actor = Actor(self.num_states, hidden_dim, self.num_actions,
                           self.max_action).to(device)
        self.actor_target = Actor(self.num_states, hidden_dim,
                                  self.num_actions, self.max_action).to(device)
        # Make copy of network
        self.actor_target.load_state_dict(self.actor.state_dict())

        ###################
        # Critic Networks #
        ###################
        self.critic = Critic(self.num_states + self.num_actions,
                             hidden_dim, self.num_actions).to(device)
        self.critic_target = Critic(self.num_states + self.num_actions,
                                    hidden_dim, self.num_actions).to(device)
        # Make copy of network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Not needed here. The replay buffer is initialized in ddpg_trainer.py
        # and handovered to the agent.
        # self.memory_buffer = MemoryBuffer(max_memory)
        self.memory_buffer = None

        #############
        # ML Set Up #
        #############
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
        # s = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # action = self.actor(s).cpu().data.numpy().flatten()

        s = torch.from_numpy(state).float()  #.unsqueeze(0)
        action = self.actor(s).cpu().data.numpy()
        return action

    def update(self, batch_size: int = 64, tau: float = None):
        """ Function to update the actor and critic components.

            Parameters:
            -----------
                batch_size: int
                    The number of samples we want to sample from the memory
                    buffer (replay buffer).
                tau: float (optional)
                    Custom tau parameter for the soft target updates.
        """

        # Make sure that the agent only starts training after sufficient
        # amount of experience
        if len(self.memory_buffer) < batch_size:
            return

        # Important: This return is suited to the ReplayBuffer in memory_buffer.py
        s, a, next_s, r, not_done = self.memory_buffer.sample(batch_size)

        # Adapted from the TD3 approach (Not vanilla DDPG).
        # Paper recommend OU Noise
        noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_s) + noise).clamp(-self.max_action, self.max_action)

        #######################
        # Q-Value Calculation #
        #######################
        q_values = self.critic(s, a)
        next_q = self.critic_target(next_s, next_action)
        q_prime = r + (self.gamma * next_q)
        critic_loss = self.critic_loss_func(q_values, q_prime)

        ####################
        # Update Critic NN #
        ####################
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ##################################
        # Actor Loss and Update Acotr NN #
        ##################################
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # If we want to use a custom tau.
        if not tau:
            tau = self.tau

        ############################################
        # Soft-Target Updates (Actor, Critic)      #
        # theta' = tau * theta + (1- tau) * theta' #
        # where theta'=target paremters            #
        ############################################
        for t_param, param in zip(self.actor_target.parameters(),
                                  self.actor.parameters()):
            update = (tau * param.data) + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)

        for t_param, param in zip(self.critic_target.parameters(),
                                  self.critic.parameters()):
            update = (tau * param.data) + ((1 - tau) * t_param.data)
            t_param.data.copy_(update)

    def run(self, env: object, steps: int = 1000, render: bool = True):
        """ Method to execute a trained agent on a domain.

            Parameters:
            -----------
                env: GymEnvironment or UnityEnvironment.
                    The environmen/domain in that the agent should interact.
                steps: int
                    Number of actions the agent should take in this run.
                render: bool
                    Flag to decide if the environment is rendered while
                    the agent is active.
        """
        state = env.reset()

        for step in range(steps):
            if render:
                env.render()
            action = self.get_action(state)
            state, reward, done, _ = env.step(action)
