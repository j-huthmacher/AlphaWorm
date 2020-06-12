import os

from td3.Actor import Actor
from td3.Critic import Critic
import torch
import numpy as np
import torch.nn.functional as F

class TD3(object):
    """Agent class that handles the training of the networks and provides outputs as actions

        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use

    """

    def __init__(self, state_dim, action_dim, max_action, env, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.env = env
        self.device = device

    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons

            Returns:
                action (float): action clipped within action range

        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):
        """Train and update actor and critic networks

            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update for main networks to target networks

            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network

        """

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            #d = torch.from_numpy(d)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))