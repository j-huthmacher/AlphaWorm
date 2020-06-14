from collections import deque
import numpy as np
import random

class ReplayMemory:

    def __init__(self, size):
        self.max_size = size
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.experience_count = 0

    def add(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        if self.experience_count >= self.max_size:
            self.buffer.popleft()
        else:
            self.experience_count += 1
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default = 1))

    def get_probabilities(self, scale):
        scaled_priorities = np.array(self.priorities ** scale)
        return scaled_priorities / sum(scaled_priorities)

    def get_importance(self, probs):
        importance = 1 / self.experience_count * 1 / probs
        importance_norm = importance / max(importance)
        return importance_norm

    # Returns a random experience
    # Batch_size = number of experiences to add
    def sample_buffer(self, batch_size, scale = 1.0):

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch_count = min(self.experience_count, batch_size)
        sample_probs = self.get_probabilities(scale)

        batch = random.choice(range(self.experience_count), k=batch_count, weights=sample_probs)

        #batch = random.sample(self.buffer, batch_count)
        for experience in batch:
            state_batch.append(experience[0])
            action_batch.append(experience[1])
            reward_batch.append(experience[2])
            next_state_batch.append(experience[3])
            done_batch.append(experience[4])

        importance = self.get_importance(sample_probs[batch])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, importance
