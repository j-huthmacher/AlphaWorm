"""
    Implementation of replay buffer (here called memory buffer) class.

    @author: j-huthmacher
"""
import random
from collections import deque
import numpy as np


class MemoryBuffer:
    """
        This class represents a memory buffer (or replay buffer)
        that can be used for RL.
    """
    def __init__(self, max_size: int):
        """ Initilaization of a memory buffer.

            Parameters:
            -----------
                max_size: int
                    Maximum size of the memory buffer.
        """

        # deque -> list-like container with fast appends and pops on either end
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.experience_count = 0



    def push(self, state: np.array, action: np.array, reward: float,
             next_state: np.array, done: bool):
        if self.experience_count >= self.max_size:
            self.buffer.popleft()
        else:
            self.experience_count += 1
        """ Function to push a entry to the memory buffer.

            Parameters:
            -----------
                state: np.array
                    Current state.
                action: np.array
                    Current action.
                reward: np.array
                    Current reward
                next_state: np.array
                    Next state.
                done: bool
                    Flag if the next state is termination state.
        """
        experience = (state, action, np.array([reward]), next_state, 1-int(done))
        self.buffer.append(experience)

    def sample(self, batch_size):
        """ Function to sample from the memory buffer.

            Parameters:
            -----------
                batch_size: int
                    Number of instances we want sample from the buffer.
            Return:
            -------
                tuple: The tuple contains arrays for each element.
                       This structure makes it possible to directly feed into
                       a neural network.
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []


        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch,
                next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
