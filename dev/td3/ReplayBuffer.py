# Expects tuples of (state, next_state, action, reward, done)
import numpy as np

class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(
            dones).reshape(-1, 1)