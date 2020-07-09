"""
    Abstract class for an RL agent.

    @author: jhuthmacher
"""
import numpy as np


class Agent(object):
    """
        Implementation of an DDPG-Agent that uses the DDPG algorithm
        for learning.
    """
    def __init__(self):
        """ Used to initialize the agents parameters.
        """

    def get_action(self, state: np.array):
        """ Returns the action that should be taken in the current state.

            Parameters:
            -----------
                state: np.array
                    The current state which is used to choose the next actions.

            Return:
            -------
                np.array: Next action to take.
        """
        raise NotImplementedError

    def update(self):
        """ Function for incporate learning in the agent.
        """
        raise NotImplementedError

    def run(self, env: object, steps: int = 100, render: bool = True):
        """ Method to execute a trained agent on a domain.

            Parameters:
            -----------
                env: UnityEnvironment or GymEnvironment

                steps: int
                    Number of actions the agent should take in this run.
                render: bool
                    Flag to decide if the environment is rendered while
                    the agent is active.
        """
        state = env.reset()

        for step in range(steps):
            if render:
                env.render()
            action = self.get_action(state)
            state, reward, done, _ = env.step(action)
