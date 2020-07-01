"""
    @author: jhuthmacher
"""
import optuna
from optuna.trial import FixedTrial
import pickle

import numpy as np
from trainer.trainer import Trainer
from agents.ddpg.ddpg import DDPGagent
from agents.ddpg.ou_noise import OUNoise
from config.config import log
from shutil import copyfile
import json

class DDPGTrainer(Trainer):
    """ Trainer class to train a DDPG agent.
    """

    def __init__(self):
        """ Initilzation
        """
        super(DDPGTrainer, self).__init__()

        self.default_trail = {
            'n_layers': 2,
            'n_units_l0': 128,
            'n_units_l1': 128,
            'actor_lr': 0.001,  # Default ADAM
            'critic_lr': 0.001,
            'gamma': 0.999,
            'tau': 0.001,
            'episodes': 2000,
            'training_steps': 1000,
            'batch_size': 512,
            'explore_threshold': 0.5
        }

    def start_training(self, env, trials: int = 1, render: bool = False,
                       name: str = None, training_steps: int = None,
                       default: bool = False):
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
        self.default = default

        # if self.default :
        #     trial = FixedTrial(self.default_trail)
        #     self.train(trial, env)
        #     return trial

        log.info(f"Optuna set up - trials: {trials}, name: {name}")
        ##########
        # Optuna #
        ##########
        study = optuna.create_study(direction="maximize")
        log.info("Start optimization!")

        study.optimize(lambda trial: self.train(trial, env, render, name,
                                                    study, training_steps),
                           n_trials=trials, timeout=600)

        log.info("Optimization done.")

        with open(f'{self.path}/{name}_study.pickle', 'wb+') as fout:
            pickle.dump(study, fout)

        log.info(f"Study stored. ({self.path}/{name}_study.pickle)")

        env.close()

        return study

    def train(self, trial: object, env: object, render: bool = False,
              name: str = None, study: object = None,
              training_steps: int = 1000):
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

        log.info(f"Start trial#{trial.number}")

        # Init directory set up.
        self.track_setup(name, trial)

        ############################
        # Hyperparameter for Agent #
        ############################
        num_hidden_layers = trial.suggest_int("n_layers", 1, 3)

        hidden_dim = []

        for i in range(num_hidden_layers):
            hidden_dim += [trial.suggest_int("n_units_l{}".format(i), 32, 256)]

        actor_lr = trial.suggest_uniform("actor_lr", 1e-4, 1e-2)
        critic_lr = trial.suggest_uniform("critic_lr", 1e-4, 1e-2)

        gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
        tau = trial.suggest_uniform("tau", 1e-4, 1e-1)

        ######################################
        # Hyperparameter for Training set up #
        ######################################
        if not training_steps:
            # Trainings steps are fixed!
            training_steps = trial.suggest_int("training_steps", 1000, 1500) # 10 - 256

        episodes = trial.suggest_int("episodes", training_steps, training_steps*1.5) # 50 - 2048
        explore_threshold = trial.suggest_uniform("explore_threshold", 0.15, 0.5) 
        batch_size = trial.suggest_int("batch_size", 32, 512)  # 10 - 256

        if self.default:
            num_hidden_layers = self.default_trail["n_layers"]
            hidden_dim = [self.default_trail["n_units_l0"],
                          self.default_trail["n_units_l1"]]

            actor_lr = self.default_trail["actor_lr"]
            critic_lr = self.default_trail["critic_lr"]

            gamma = self.default_trail["gamma"]
            tau = self.default_trail["tau"]

            episodes = self.default_trail['episodes']
            training_steps = self.default_trail['training_steps']

            explore_threshold = self.default_trail['explore_threshold']
            batch_size = self.default_trail['batch_size']

            log.info(f"Params {self.default_trail}")
        else:
            log.info(f"Params {trial.params}")

        ################
        # Create agent #
        ################
        self.ddpg_agent = DDPGagent(env, hidden_dim, actor_lr, critic_lr,
                                    gamma, tau)

        rewards = []
        noise = OUNoise(env.action_space)

        log.info(f"Start episodes ({episodes}) with {training_steps} steps.")

        self.episode_num = 0

        overall_steps = 0
        for episode in range(episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            if episode % 100 == 0:
                log.info(f"Trainings-Step: {episode}/{episodes}")

            for step in range(training_steps):
                if render:
                    env.render()

                # Explorate the first
                if overall_steps < explore_threshold * (training_steps * episodes):
                    if step % 100 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps} (Explore)")

                    action = env.action_space.sample()
                else:
                    if step % 100 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps}")

                    action = self.ddpg_agent.get_action(state)
                    action = noise.get_action(action, step)

                new_state, reward, done, _ = env.step(action)
                self.ddpg_agent.memory_buffer.push(state, action, reward,
                                                   new_state, done)

                if len(self.ddpg_agent.memory_buffer) > batch_size:
                    # Train/Update the agent's parameters
                    self.ddpg_agent.update(batch_size)

                state = new_state
                episode_reward += reward

                self.track_training_reward(episode_reward,
                                           step,
                                           training_steps)

                if done:
                    self.track_successful_episodes(episode,
                                                   episode_reward,
                                                   step)
                    state, done = env.reset(), False
                    episode_reward = 0
                    # episode_timesteps = 0
                    self.episode_num += 1
                
                overall_steps += 1

            self.track_reward(episode_reward, episode)
            trial.report(episode_reward, episode)

            rewards.append(episode_reward)

        log.info("End episode!")

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Save a trained model to a file.
        # Think if storing every model is necessary
        if study and name:
            try:
                if study.best_value < np.array(rewards).mean():
                    with open(f'{self.path}/best_agent/{name}_best_agent.pickle', 'wb+') as fout:
                        pickle.dump(self.ddpg_agent, fout)

                    if self.default:
                        with open(f'{self.path}/best_agent/{name}_best_params.json', 'w+') as fout:
                            json.dump(self.default_trail, fout)
                    else:
                        with open(f'{self.path}/best_agent/{name}_best_params.json', 'w+') as fout:
                            json.dump(trial.params, fout)

                    self.training_rewards_df.to_csv(f'{self.path}/best_agent/trainings_rewards.csv')

                    copyfile(f'{self.path_trial}/results/rewards.csv', f'{self.path}/best_agent/rewards.csv')

                    copyfile(f'{self.path_trial}/results/successful_episodes.csv', f'{self.path}/best_agent/successful_episodes.csv')

                    log.info(f"Best agent stored at {self.path}/best_agent/{name}_best_agent.pickle")
            except Exception:
                # Is executed when no study exists. I.e. first run.
                with open(f'{self.path}/best_agent/{name}_best_agent.pickle', 'wb+') as fout:
                    pickle.dump(self.ddpg_agent, fout)

                if self.default:
                    with open(f'{self.path}/best_agent/{name}_best_params.json', 'w+') as fout:
                        json.dump(self.default_trail, fout)
                else:
                    with open(f'{self.path}/best_agent/{name}_best_params.json', 'w+') as fout:
                        json.dump(trial.params, fout)

                self.training_rewards_df.to_csv(f'{self.path}/best_agent/trainings_rewards.csv')

                copyfile(f'{self.path_trial}/results/rewards.csv', f'{self.path}/best_agent/rewards.csv')

                copyfile(f'{self.path_trial}/results/successful_episodes.csv', f'{self.path}/best_agent/successful_episodes.csv')

                log.info(f"Initial agent stored at {self.path}/best_agent/{name}_best_agent.pickle")

        log.info(f"End trial#{trial.number}")

        return np.array(rewards).mean()
