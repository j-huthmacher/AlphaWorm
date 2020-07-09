"""
    Abstract trainer class for a unified interface.

    @author: jhuthmacher
"""

import optuna
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class Trainer(object):
    """ Abstract trainer object for bundle trainings procedures
    """

    def __init__(self):
        """ Initilization
        """
        self.training_rewards_df = pd.DataFrame([])
        self.action_history_df = pd.DataFrame([])
        self.action_history = []
        self.training_rewards = []
        self.curr_episode = 0
        self.successful_episdes = {}

    def train(self, env: object, render: bool = False, name: str = None):
        """ Standard train method for train an DDPG agent on an environment.

            Parameters:
            -----------
                env: GymEnvironment or UnityEnvironment
                    The environment that is used for training
                render: bool (optional)
                    For enable the rendering during training. Only usable
                    for gym environments.
                name: str
                    Name of the agents for storing the results etc.
        """
        raise NotImplementedError

    def track_setup(self, model_name: str, trial: optuna.Trial = None):
        """ Initialization of the tracking et up. I.e. create folders and
            save paths as class variables.

            Parameters:
            -----------
                model_name: str
                    Name of the model/agent you currently train.
                trial: optuna.Trial
                    Trial from HPO for tracking each trial separately.
        """
        self.path = f'models/{datetime.now().date()}/{model_name}'
        self.path_trial = self.path
        if not trial:
            self.path_trial = f'models/{datetime.now().date()}/{model_name}'
        else:
            self.path_trial = f'models/{datetime.now().date()}/{model_name}/{trial.number}_{model_name}'

        # Create general directory
        folder = Path(self.path)
        folder.mkdir(parents=True, exist_ok=True)

        # Create general directory
        folder = Path(self.path_trial)
        folder.mkdir(parents=True, exist_ok=True)

        # Create result directory
        folder = Path(self.path_trial + "/results")
        folder.mkdir(parents=True, exist_ok=True)

        # Create result directory
        folder = Path(self.path + "/best_agent")
        folder.mkdir(parents=True, exist_ok=True)

        # Init reqard csv with correct header
        with open(f'{self.path_trial}/results/rewards.csv', 'w+') as f:
            f.write("reward,episode")

        # Init reqard csv with correct header
        with open(f'{self.path_trial}/results/successful_episodes.csv', 'w+') as f:
            f.write("episode,reqard,reward,training_step")

    def track_action(self, action: np.ndarray, step: int, num_steps: int):
        """ Tracking actions for creating action history.

            Parameters:
            -----------
                action: np.ndarray
                    Action that should be tracked.
                step: int
                    The current step in which the action taked place
                    (e.g. current trainings step)
                num_steps: int
                    Overall number of steps.
        """
        self.action_history.append(action)

        if step == num_steps - 1:
            self.action_history_df[f"Episode {self.curr_episode}"] = self.action_history
            self.action_history_df.to_csv(f'{self.path_trial}/results/action_history.csv')
            self.action_history = []
            self.curr_episode += 1

    def track_reward(self, reward: float, episode: int):
        """ Tracking rewards.

            Parameters:
            -----------
                reward: float
                    Reward for the corresponding episode.
                episode: int
                    Current episode to which the reward corresponds to.
        """
        with open(f'{self.path_trial}/results/rewards.csv', 'a') as f:
            f.write(f"\n{reward},{episode}")

    def track_training_reward(self, reward: float, step: int, num_steps: int):
        """ Tracking the reward during training.

            Parameters:
            -----------
                reward: float
                    The reward to the corresponding training step.
                step: int
                    The current training step to which the reward corresponds to.
                num_steps: int
                    Overall number of steps.
        """
        self.training_rewards.append(reward)

        if step == num_steps - 1:
            self.training_rewards_df[f"Episode {self.curr_episode}"] = self.training_rewards
            self.training_rewards_df.to_csv(f'{self.path_trial}/results/trainings_rewards.csv')
            self.training_rewards = []
            self.curr_episode += 1

    def track_successful_episodes(self, episode: int, reward: float, training_step: int):
        """ Tracking successfull episodes.

            Parameters:
            -----------
                episode: int
                    Curreent episode.
                reward: float
                    Reward corresponding to the current episode.
                training_step: int
                    The training steps in which the agent reached the goal.
        """
        with open(f'{self.path_trial}/results/successful_episodes.csv', 'a') as f:
            f.write(f"\n{episode},{reward},{training_step}")
