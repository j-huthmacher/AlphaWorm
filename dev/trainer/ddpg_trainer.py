"""
    @author: jhuthmacher
"""
import optuna
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from trainer.trainer import Trainer
from agents.ddpg.ddpg import DDPGagent
from agents.ddpg.ou_noise import OUNoise


class DDPGTrainer(Trainer):
    """ Trainer class to train a DDPG agent.
    """

    def __init__(self):
        """ Initilzation
        """
        self.path = f'models/{datetime.now().date()}'
        folder = Path(self.path)
        folder.mkdir(parents=True, exist_ok=True)

    def start_training(self, env, trials: int = 1, render: bool = False,
                       name: str = None):
        """ Method to start the training. Conatains HPO configurations

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

        ##########
        # Optuna #
        ##########
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.train(trial, env, render, name, study),
                       n_trials=trials, timeout=600)

        with open(f'{self.path}/{name}_study.pickle', 'wb+') as fout:
            pickle.dump(study, fout)

        env.close()

        return study

    def train(self, trial: object, env: object, render: bool = False,
              name: str = None, study: object = None):
        """ Implementation of the training procedure for ddpg.

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

        ############################
        # Hyperparameter for Agent #
        ############################
        num_hidden_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dim = []

        for i in range(num_hidden_layers):
            hidden_dim += [trial.suggest_int("n_units_l{}".format(i), 64, 512)]

        actor_lr = trial.suggest_uniform("actor_lr", 1e-6, 1e-1)
        critic_lr = trial.suggest_uniform("critic_lr", 1e-6, 1e-1)

        gamma = trial.suggest_uniform("gamma", 0.001, 0.999)
        tau = trial.suggest_uniform("tau", 1e-6, 1e-1)

        ######################################
        # Hyperparameter for Training set up #
        ######################################
        episodes = trial.suggest_int("episodes", 50, 2048) # 50 - 2048
        training_steps = trial.suggest_int("training_steps", 50, 512) # 10 - 256
        batch_size = trial.suggest_int("batch_size", 10, 256) # 10 - 256

        ################
        # Create agent #
        ################
        ddpg_agent = DDPGagent(env, hidden_dim, actor_lr, critic_lr,
                               gamma, tau)


        np.random.seed(0)

        rewards = []

        noise = OUNoise(mu = np.zeros(env.action_space))

        for episode in range(episodes):
            state = env.reset()
            #noise.reset()
            episode_reward = 0

            for step in range(training_steps):
                if render:
                    env.render()
                action = ddpg_agent.get_action(state)
                action = noise.get_action(action, step)
                new_state, reward, done, _ = env.step(action)
                ddpg_agent.memory_buffer.push(state, action, reward,
                                              new_state, done)

                if len(ddpg_agent.memory_buffer) > batch_size:
                    ddpg_agent.update(batch_size)

                state = new_state
                episode_reward += reward

            trial.report(episode_reward, episode)

            rewards.append(episode_reward)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Save a trained model to a file.
        # Think if storing every model is necessary
        if study and name:
            try:
                if study.best_value < np.array(rewards).mean():
                    with open(f'{self.path}/{name}.pickle', 'wb+') as fout:
                        pickle.dump(ddpg_agent, fout)
            except Exception:
                # Is executed when no study exists. I.e. first run.
                with open(f'{self.path}/{name}.pickle', 'wb+') as fout:
                    pickle.dump(ddpg_agent, fout)

        return np.array(rewards).mean()
