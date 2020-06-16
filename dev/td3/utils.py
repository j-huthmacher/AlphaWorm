import os
import time

import numpy as np
import torch
import json

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class ReplayBufferHandler(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_size = max_size
		self.best_list = []
		self.max_reward = float('-inf')
		self.ptr = 0
		self.size = 0

	def add(self, buffer):
		if np.average(buffer.reward) > self.max_reward:
			self.best_list.append(buffer)
			self.max_reward = np.average(buffer.reward)
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)

	def save(self, folder='buffers'):
		states = np.empty((self.max_size, self.state_dim))
		actions = np.empty((self.max_size, self.action_dim))
		next_states = np.empty((self.max_size, self.state_dim))
		rewards = np.empty((self.max_size, 1))
		not_dones = np.empty((self.max_size, 1))
		for list_index in range(len(self.best_list)):
			np.append(states, self.best_list[list_index].state, axis=0)
			np.append(actions, self.best_list[list_index].action, axis=0)
			np.append(next_states, self.best_list[list_index].next_state, axis=0)
			np.append(rewards, self.best_list[list_index].reward, axis=0)
			np.append(not_dones, self.best_list[list_index].not_done, axis=0)

		self.save_numpy(folder, 'state', states)
		self.save_numpy(folder, 'action', actions)
		self.save_numpy(folder, 'next_state', next_states)
		self.save_numpy(folder, 'reward', rewards)
		self.save_numpy(folder, 'not_dones', not_dones)


	def save_numpy(self, folder, filename, array):
		np.save(f"./{folder}/{filename}_{int(round(time.time() * 1000))}", array)

	def load(self, folder='buffers'):
		replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

		states = np.empty((self.max_size, self.state_dim))
		actions = np.empty((self.max_size, self.action_dim))
		next_states = np.empty((self.max_size, self.state_dim))
		rewards = np.empty((self.max_size, 1))
		not_dones = np.empty((self.max_size, 1))

		for filename in os.listdir(folder):
			if filename.startswith("state"):
				np.append(states, np.load(f"./{folder}/{filename}"), axis=0)
			if filename.startswith("action"):
				np.append(actions, np.load(f"./{folder}/{filename}"), axis=0)
			if filename.startswith("next_state"):
				np.append(next_states, np.load(f"./{folder}/{filename}"), axis=0)
			if filename.startswith("reward"):
				np.append(rewards, np.load(f"./{folder}/{filename}"), axis=0)
			if filename.startswith("done"):
				np.append(not_dones, np.load(f"./{folder}/{filename}"), axis=0)

		for index in range(self.max_size):
			replay_buffer.add(states[index], actions[index], next_states[index], rewards[index], not_dones[index])

		return replay_buffer
