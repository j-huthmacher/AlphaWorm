"""
    Ornstein-Ulhenbeck Process for creating the noise.

    Source: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""




import numpy as np


class OUNoise(object):
    """
        This strategy implements the Ornstein-Uhlenbeck process, which adds
        time-correlated noise to the actions taken by the deterministic policy.
        The OU process satisfies the following stochastic differential equation:
        dxt = theta*(mu - xt)*dt + sigma*dWt
        where Wt denotes the Wiener process
        Based on the rllab implementation.
    """

    def __init__(self, action_space, mu, theta=0.15,
                 max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()


    def reset(self):
        self.state = np.zeros(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = (self.theta * (self.mu - x) + self.sigma *
              np.random.randn(self.action_dim))
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = (self.max_sigma - (self.max_sigma - self.min_sigma) *
                      min(1.0, t / self.decay_period))

        return np.clip(action + ou_state, self.low, self.high)
