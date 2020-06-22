import os
import shutil
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
	def __init__(self, state_dim, action_dim, der_size=int(1), max_size=int(1e6)):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.der_size = der_size
		self.max_size = max_size
		self.best_list = []
		self.max_reward = float('-inf')
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
		folder_name = f"./{folder}/tmp/"
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)

		states = None
		actions = None
		next_states = None
		rewards = None
		not_dones = None

		for list_index in range(len(self.best_list)):
			if states is None:
				states = self.best_list[list_index].state
			else:
				states = np.append(states, self.best_list[list_index].state, axis=0)
			if actions is None:
				actions = self.best_list[list_index].action
			else:
				actions = np.append(actions, self.best_list[list_index].action, axis=0)
			if next_states is None:
				next_states = self.best_list[list_index].next_state
			else:
				next_states = np.append(next_states, self.best_list[list_index].next_state, axis=0)
			if rewards is None:
				rewards = self.best_list[list_index].reward
			else:
				rewards = np.append(rewards, self.best_list[list_index].reward, axis=0)
			if not_dones is None:
				not_dones = self.best_list[list_index].not_done
			else:
				not_dones = np.append(not_dones, self.best_list[list_index].not_done, axis=0)

		self.save_numpy(folder_name, 'state', states)
		self.save_numpy(folder_name, 'action', actions)
		self.save_numpy(folder_name, 'next_state', next_states)
		self.save_numpy(folder_name, 'reward', rewards)
		self.save_numpy(folder_name, 'not_done', not_dones)
		shutil.make_archive(f"./{folder}/episode_{int(round(time.time() * 1000))}", 'zip', folder_name)
		if os.path.exists(folder_name):
			shutil.rmtree(folder_name, ignore_errors=True)
		self.best_list = []
		self.size = 0

	def save_numpy(self, folder, filename, array):
		np.save(f"./{folder}/{filename}", array)

	def load(self, folder='buffers', batch_load=False, batch_size=1):
		states = None
		actions = None
		next_states = None
		rewards = None
		not_dones = None

		if not os.path.exists(f"./{folder}/"):
			print("No Buffer; No Folder")
			return

		if batch_load:
			file_list = [a_file for a_file in os.listdir(folder) if a_file.endswith('.zip')]
			if len(file_list) > 0:
				select = min(batch_size, len(file_list))
				file_list = np.random.choice(file_list, select)
				print('Loading Buffers: ', file_list)
			else:
				print("No Buffer; No Files in Folder")
				return
		else:
			file_list = os.listdir(folder)

		for archive_files in file_list:
			print("Loading:" + archive_files)
			tmp_folder = f"{folder}/tmp/"
			shutil.unpack_archive(f"./{folder}/{archive_files}", f"./{tmp_folder}", "zip")
			for filename in os.listdir(f"./{tmp_folder}/"):
				if filename.startswith("state"):
					if states is None:
						states = np.load(f"./{tmp_folder}/{filename}")
					else:
						states = np.append(states, np.load(f"./{tmp_folder}/{filename}"), axis=0)
				if filename.startswith("action"):
					if actions is None:
						actions = np.load(f"./{tmp_folder}/{filename}")
					else:
						actions = np.append(actions, np.load(f"./{tmp_folder}/{filename}"), axis=0)
				if filename.startswith("next_state"):
					if next_states is None:
						next_states = np.load(f"./{tmp_folder}/{filename}")
					else:
						next_states = np.append(next_states, np.load(f"./{tmp_folder}/{filename}"), axis=0)
				if filename.startswith("reward"):
					if rewards is None:
						rewards = np.load(f"./{tmp_folder}/{filename}")
					else:
						rewards = np.append(rewards, np.load(f"./{tmp_folder}/{filename}"), axis=0)
				if filename.startswith("not_done"):
					if not_dones is None:
						not_dones = np.load(f"./{tmp_folder}/{filename}")
					else:
						not_dones = np.append(not_dones, np.load(f"./{tmp_folder}/{filename}"), axis=0)
			if os.path.exists(tmp_folder):
				shutil.rmtree(tmp_folder, ignore_errors=True)

		replay_buffer = ReplayBuffer(states.shape[1], actions.shape[1], max_size=states.shape[0])
		for index in range(self.max_size):
			replay_buffer.add(states[index], actions[index], next_states[index], rewards[index], not_dones[index])
		print('Loading Buffer done')
		return replay_buffer
