from collections import deque
import numpy as np
import random

# Set a = 0 for normal experience replay

"""
Implement PER:


1. Select Prioritizing Replay Buffer
2. Get TD-Error from update function
3. Save TD-Error in Buffer
4. Get TD-Error when sampling 
5. Update self.priorities with the corresponding td_errprs



Greedy TD-Priorization makes NN overfit easily 

 => trade-off between greedy TD-error prioritization and pure random sampling

"""

class MemoryBuffer:

    def __init__(self, size):
        self.max_size = size
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.experience_count = 0

    def push(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        if self.experience_count >= self.max_size:
            self.buffer.popleft()
        else:
            self.experience_count += 1.0
        self.buffer.append(experience)
        self.priorities.append(1.0)

    def get_probabilities(self, priority_scale):
        scaled_priorities = list(np.array(self.priorities) ** priority_scale)
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probs):
        importance = 1 / self.experience_count * 1 / probs
        importance_norm = importance / max(importance)
        return importance_norm

    # Returns a random experience
    # Batch_size = number of experiences to add
    # Scale: 0=Random Sampling
    def sample(self, batch_size, scale = 0.5):


        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        importance_batch = []

        sample_size = min(len(self.buffer), batch_size)
        probs = self.get_probabilities(scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=probs)
        samples = np.array(self.buffer)[sample_indices]
        #importance = self.get_importance(probs[sample_indices])

        for index, experience in enumerate(self.buffer):
            if index in sample_indices:
                state_batch.append(experience[0])
                action_batch.append(experience[1])
                reward_batch.append(experience[2])
                next_state_batch.append(experience[3])
                done_batch.append(experience[4])
                importance_batch.append(self.priorities[index])


        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, importance_batch, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for index, error in zip(indices, errors):
            td_error = error.detach().numpy()[0]
            self.priorities[index] = abs(td_error) + offset
