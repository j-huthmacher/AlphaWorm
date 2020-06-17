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


class DynamicExperienceReplay(object):
	def __init__(self, state_dim, action_dim, der_size=int(1e1), max_size=int(1e6)):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.der_size = der_size
		self.max_size = max_size
		self.best_list = []
		self.max_reward = float('-inf')
		self.folder_number = 0
		self.size = 0

	def add(self, buffer):
		average_reward = np.average(buffer.reward)
		changed = False

		if self.size >= self.der_size:
			min_reward = np.average(self.best_list[-1].reward)
			if average_reward > min_reward:
				self.best_list.pop()
				self.best_list.append(buffer)
				changed = True
		else:
			self.best_list.append(buffer)
			self.size = min(self.size + 1, self.der_size)
			changed = True

		if (self.size >= self.der_size - 1) and changed:
			self.best_list.sort(key=lambda x: np.average(x.reward), reverse=True)

	def save(self, folder='buffers'):
		folder_name = f"./{folder}/episode_{int(round(time.time() * 1000))}"
		os.makedirs(folder_name)
		self.folder_number += 1

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

		self.save_numpy(folder_name, 'state', states)
		self.save_numpy(folder_name, 'action', actions)
		self.save_numpy(folder_name, 'next_state', next_states)
		self.save_numpy(folder_name, 'reward', rewards)
		self.save_numpy(folder_name, 'not_done', not_dones)


	def save_numpy(self, folder, filename, array):
		np.save(f"./{folder}/{filename}", array)

	def load(self, folder='buffers'):
		replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

		states = None
		actions = None
		next_states = None
		rewards = None
		not_dones = None

		for dirpath, dirnames, files in os.walk(folder):
			for dirname in dirnames:
				for filename in os.listdir(f"./{folder}/{dirname}"):
					if filename.startswith("state"):
						if states is None:
							states = np.load(f"./{folder}/{dirname}/{filename}")
						else:
							np.append(states, np.load(f"./{folder}/{dirname}/{filename}"), axis=0)
					if filename.startswith("action"):
						if actions is None:
							actions = np.load(f"./{folder}/{dirname}/{filename}")
						else:
							np.append(actions, np.load(f"./{folder}/{dirname}/{filename}"), axis=0)
					if filename.startswith("next_state"):
						if next_states is None:
							next_states = np.load(f"./{folder}/{dirname}/{filename}")
						else:
							np.append(next_states, np.load(f"./{folder}/{dirname}/{filename}"), axis=0)
					if filename.startswith("reward"):
						if rewards is None:
							rewards = np.load(f"./{folder}/{dirname}/{filename}")
						else:
							np.append(rewards, np.load(f"./{folder}/{dirname}/{filename}"), axis=0)
					if filename.startswith("not_done"):
						if not_dones is None:
							not_dones = np.load(f"./{folder}/{dirname}/{filename}")
						else:
							np.append(not_dones, np.load(f"./{folder}/{dirname}/{filename}"), axis=0)

		for index in range(self.max_size):
			replay_buffer.add(states[index], actions[index], next_states[index], rewards[index], not_dones[index])

		return replay_buffer
