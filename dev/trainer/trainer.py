"""
    Abstract trainer class for a unified interface.

    @author: jhuthmacher
"""

from optuna import Trial, Study
from pathlib import Path
from datetime import datetime
import pandas as pd

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

    def start_training(env: any, trials: int = 1, render: bool = False,
                       name: str = None):
        """ Function to exectue the individual training function.

            Parameters:
            -----------
                env: UnityEnvironment or GymEnvironment
                    The environment the agent interacts in. Could be a Unity
                    environment wrapped in a gym environment or directly a
                    gym environment.
                trials: int
                    Number of training runs that are executed from the HPO
                    library (trials = 2 means the train method is two times
                    executed with different parameters)
                render: bool
                    Flag to decide if we want to render in case of a gym
                    environment.
                name: str
                    A name for the model/agent that is used to store the best
                    model.
        """
        raise NotImplementedError

    def train_hpo(trial: Trial, env: any, render: bool = False, name: str = None,
              study: Study = None):
        """ Individual train function that implement the actual training procedure.

            Parameters:
            -----------
                trial: optuna.Trial
                    The trail object from the handovered from the HPO library.
                    This contains functionality for selecting hyper parameters.
                env: UnityEnvironment or GymEnvironment
                    The environemnt in which the agent should train.
                render: bool
                    Falg to decide if we want to render the steps in case of a
                    gym environment.
                name: str
                    A name for the model/agent that is used to store the best
                    model.
                study: optuna.Study
                    Study object that contains informationen about the training
                    (parameters for each run, best parameter set etc.)
            Return:
            -------
                reward: float
                    Since we want to optimize the reward in our case we return
                    the reward of the trainings run.
        """
        raise NotImplementedError

    def track_setup(self, model_name: str, trial=None):
        """
        """
        self.path = f'models/{datetime.now().date()}/{model_name}'
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

    def track_action(self, action, step, num_steps):
        """
        """
        self.action_history.append(action)

        if step == num_steps - 1:
            self.action_history_df[f"Episode {self.curr_episode}"] = self.action_history
            self.action_history_df.to_csv(f'{self.path_trial}/results/action_history.csv')
            self.action_history = []
            self.curr_episode += 1

    def track_reward(self, reward, episode):
        """
        """
        with open(f'{self.path_trial}/results/rewards.csv', 'a') as f:
            f.write(f"\n{reward},{episode}")

    def track_training_reward(self, reward, step, num_steps):
        """
        """
        self.training_rewards.append(reward)

        if step == num_steps - 1:
            self.training_rewards_df[f"Episode {self.curr_episode}"] = self.training_rewards
            self.training_rewards_df.to_csv(f'{self.path_trial}/results/trainings_rewards.csv')
            self.training_rewards = []
            self.curr_episode += 1

    def track_successful_episodes(self, episode, reward, training_step):
        """
        """
        with open(f'{self.path_trial}/results/successful_episodes.csv', 'a') as f:
            f.write(f"\n{episode},{reward},{training_step}")
