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
		self.best_list = [None] * self.max_size
		self.max_reward = float('-inf')
		self.ptr = 0
		self.size = 0

	def add(self, buffer):
		print(self.ptr)
		if np.average(buffer.reward) > self.max_reward:
			self.best_list[self.ptr] = buffer
			self.max_reward = np.average(buffer.reward)
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)

	def save(self, filename='replay_buffer_handler.json'):
		print(self.best_list)
		with open(filename, 'w') as outfile:
			json.dump(self.best_list, outfile)

	def load(self, filename='replay_buffer_handler.json'):
		with open(filename) as json_file:
			self.best_list = json.load(json_file)

		replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
		for list_index in range(len(self.best_list)):
			for buffer_index in range(self.best_list[list_index].size):
				state = self.best_list[list_index].state[buffer_index]
				action = self.best_list[list_index].action[buffer_index]
				next_state = self.best_list[list_index].next_state[buffer_index]
				reward = self.best_list[list_index].reward[buffer_index]
				done = self.best_list[list_index].not_done[buffer_index]
				replay_buffer.add(state, action, next_state, reward, done)

		return replay_buffer
