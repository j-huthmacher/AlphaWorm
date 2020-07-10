""" Implementation of trainer that trains the DDPG agent.

    This class is responsible for the whole training procedure.

    @author: jhuthmacher
"""
#######
# HPO #
#######
import optuna

####################
# Default Packages #
####################
from shutil import copyfile
import json
from collections import deque
from datetime import datetime
import pickle
from pathlib import Path

####################
# ML / RL Packages #
####################
import numpy as np
import pandas as pd
import torch
from gym import wrappers

from trainer import Trainer
from agents import DDPGagent, OUNoise, GaussianNoise, ReplayBuffer
from config.config import log


class DDPGTrainer(Trainer):
    """ Trainer class to train a DDPG agent.
    """

    def __init__(self):
        """ Initilzation
        """
        super(DDPGTrainer, self).__init__()

        # Legacy. Needed for the HPO implementation
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

        #################################
        # Normal Training configuration #
        #################################
        self.config = {
            'n_layers': 2,  # Num of layers used for Actor and Critic. Not used!
            'n_units_l0': 128,  # Dim of the first hidden layer in the Actor and Critic net
            'n_units_l1': 128,  # Dim of the second hidden layer in the Actor and Critic net
            'actor_lr': 0.001,  # Default ADAM
            'critic_lr': 0.001,
            'gamma': 0.999,
            'tau': 0.0001,
            'episodes': 2000,
            'training_steps': 1000,
            'batch_size': 256,  # Sample size of the replay buffer
            'explore_threshold': 0.5,  # Define how long the agent should explore relative to all steps
            'evaluation_steps': 10,
            'evaluation_lim': None  # An evaluation limit that can be set if we don't want to wait until a episode is done.
        }

    def train(self, env: object, render: bool = False, name: str = None,
              render_training: bool = False):
        """ Standard train method for train an DDPG agent on an environment.

            Parameters:
            -----------
                env: GymEnvironment or UnityEnvironment
                    The environment that is used for training
                render: bool (optional)
                    For enable the rendering during training. Only usable for gym environments.
                name: str
                    Name of the agents for storing the results etc.
        """
        #############
        # Set seeds #
        #############
        env.action_space.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        #########################
        # Init directory Set Up #
        #########################
        self.track_setup(name)

        ###########################
        # Gather env. information #
        ###########################
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        ################
        # Create agent #
        ################
        self.ddpg_agent = DDPGagent(env,
                                    [
                                        self.config["n_units_l0"],
                                        self.config["n_units_l1"]
                                    ],
                                    self.config["actor_lr"],
                                    self.config["critic_lr"],
                                    self.config["gamma"],
                                    self.config["tau"])

        self.ddpg_agent.max_action = max_action

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        rewards = []
        noise = OUNoise(env.action_space)

        episodes = self.config["episodes"]
        training_steps = self.config["training_steps"]
        explore_threshold = self.config["explore_threshold"]
        batch_size = self.config["batch_size"]

        log.info(f"Start episodes ({episodes}) with {training_steps} steps.")

        self.episode_num = 0
        self.eval_rewards = []
        self.train_rewards = []

        log.info("Parameter:")
        log.info(self.config)

        overall_steps = 0

        #############################
        # Start Trainings Procedure #
        #############################
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            if episode % 10 == 0:
                log.info(f"Episode-Step: {episode}/{episodes}")

            ############
            # Training #
            ############
            for step in range(training_steps):

                ###############
                # Exploration #
                ###############
                if overall_steps < explore_threshold * (training_steps * episodes):
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps} (Explore)")

                    # Just sample some action (random sampling)
                    action = env.action_space.sample()
                else:
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps}")

                    action = self.ddpg_agent.get_action(np.array(state))

                    # Gaussian Noise. Used from TD3. Paper recommend OU Noise
                    # noise = np.random.normal(0, self.ddpg_agent.max_action * 0.1, size=self.ddpg_agent.num_actions)

                    # action = (action + noise).clip(-self.ddpg_agent.max_action,
                    #                                self.ddpg_agent.max_action)
                    action = noise.get_action(action, step)


                # Important for the Pendulum domain.
                if np.array(action).size > 1:
                    action = np.array(action).reshape((1, 9))

                # Tracking to files
                # self.track_action(action, step,
                #                   training_steps)

                next_state, reward, done, _ = env.step(action)

                if render_training:
                    env.render()

                # From TD3 implementation
                # done = (True
                #         if step < self.config["training_steps"]
                #         else False)
                done_bool = float(done)

                # Gather experiences
                replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

                # Tracking to files
                # self.track_training_reward(episode_reward,
                #                            step,
                #                            training_steps)

                if done:
                    self.track_successful_episodes(episode,
                                                   episode_reward,
                                                   step)
                    state, done = env.reset(), False
                    episode_reward = 0
                    self.episode_num += 1

                overall_steps += 1

            state, done = env.reset(), False
            episode_reward = 0

            # Important: Set an appropriate replay buffer.
            self.ddpg_agent.memory_buffer = replay_buffer

            #################################
            # Update neural nets (Learning) #
            #################################
            if (len(self.ddpg_agent.memory_buffer) > batch_size and
                overall_steps >= explore_threshold * (training_steps * episodes)):
                for step in range(training_steps):
                    # Only train the nets when we have enough experience and we
                    # do not randomly explore anymore.
                    self.ddpg_agent.update(batch_size)

            ########################
            # Evaluation per epoch #
            ########################
            # Evaluation after we learned something
            if overall_steps >= explore_threshold * (training_steps * episodes):
                log.info(f"Start Evaluation: {self.config['evaluation_steps']}")

                self.eval_episode_reward = 0

                # For monitoring the progress.
                if render:
                    path = f'models/{datetime.now().date()}/{name}/'
                    eval_env = wrappers.Monitor(env, path, force=True)
                else:
                    eval_env = env

                # env.action_space.seed(0)
                for step in range(self.config["evaluation_steps"]):
                    if step % 50 == 0:
                        log.info(f"Evaluation-Episode: {step}/{self.config['evaluation_steps']}")

                    state = eval_env.reset()
                    done = False
                    k = 0
                    while not done:
                        action = self.ddpg_agent.get_action(np.array(state))

                        # For pendulum
                        if np.array(action).size > 1:
                            action = np.array(action).reshape((1, 9))

                        action = action.clip(-self.ddpg_agent.max_action,
                                            self.ddpg_agent.max_action)

                        state, reward, done, _ = eval_env.step(action)

                        self.eval_episode_reward += reward

                        # If you want to do not all evaluation steps. Important
                        # for the gym monitor it is important that the episode
                        # is done before the environment is closed!
                        if self.config["evaluation_lim"] is not None and self.config["evaluation_lim"] < k:
                            break

                        k += 1

                if self.config['evaluation_steps'] > 0:
                    log.info(f"Evaluation Reward: {self.eval_episode_reward/self.config['evaluation_steps']}")
                
                if self.config["evaluation_steps"] > 0:
                    self.eval_rewards.append(self.eval_episode_reward/self.config["evaluation_steps"])

            # Loacal tracking
            # self.track_reward(episode_reward, episode)

            rewards.append(episode_reward)

            ################################
            # Persist Tracking per Episode #
            ################################
            folder = Path(f'models/{datetime.now().date()}/{name}/')
            folder.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(self.eval_rewards).to_csv(f'models/{datetime.now().date()}/{name}/eval_rewards.csv')

            # Training rewards
            # pd.DataFrame(self.train_rewards).to_csv(f'models/{datetime.now().date()}/{name}/train_rewards.csv')

            with open(f'models/{datetime.now().date()}/{name}/config.json', "w+") as f:
                json.dump(self.config, f)

            ##############
            # Save Agent #
            ##############
            with open(f'models/{datetime.now().date()}/{name}/ddpg_agent_training.pickle', "wb+") as f:
                pickle.dump(self.ddpg_agent, f)

        log.info("End episode!")
        log.info("Close environment")
        env.close()

    def train_baseline(self, env: object, name: str, render: bool = False,
                       nb_epochs: int = 50, nb_epoch_cycles: int = 20,
                       nb_rollout_steps: int = 100, nb_train_steps: int = 100,
                       nb_eval_steps: int = 100):
        """ Trainings procedure from baseline implementation

            This implementation has some additional loops and parameters.
            We do not recommend to use this training procedure, since we
            are not sure if it works properly. It was implemented for testing
            purpose.

            Link: https://github.com/openai/baselines/tree/master/baselines/ddpg

            Parameters:
            -----------
                env: GymEnvironment or UnityEnvironment
                    THe environment in that the agent interact.
                name: str
                    Name of the agent for tracking purpose.
                render: bool
                    Render flag that decide to render the steps.
                nb_epochs: int
                    Number of epochs (corresponds to number of episodes)
                nb_epoch_cycles: int
                    Number of cycles within one epoch.
                nb_rollout_steps: int
                    Number of steps the agent should do for exploration and explotation.
                    (Within an epoch cycle)
                nb_train_steps: int
                    Number of trainings steps the agent should do.
                    (Within an epcoh cycle)
                nb_eval_steps: int
                    Number of evaluation steps.
        """

        ###########################
        # Gather env. information #
        ###########################
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        ################
        # Create agent #
        ################
        self.ddpg_agent = DDPGagent(env,
                                    [
                                        self.config["n_units_l0"],
                                        self.config["n_units_l1"]
                                    ],
                                    self.config["actor_lr"],
                                    self.config["critic_lr"],
                                    self.config["gamma"],
                                    self.config["tau"])
        self.ddpg_agent.max_action = max_action

        # Use some white noise.
        gaussian_noise = GaussianNoise(mu=np.zeros(self.ddpg_agent.num_actions),
                                       sigma=np.ones(self.ddpg_agent.num_actions))

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        self.episode_reward = np.zeros(1, dtype=np.float32)  # vector
        self.episode_step = np.zeros(1, dtype=int)  # vector
        self.episodes = 0  # scalar
        t = 0  # scalar

        epoch = 0
        batch_size = self.config["batch_size"]

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_actions = []
        self.epoch_reward = []
        self.epoch_episodes = 0
        self.episode_rewards_history = deque(maxlen=100)

        self.eval_rewards = []
        self.train_rewards = []
        self.eval_episode_rewards = []

        # Number of total iterations the agent takes
        self.number_it = (nb_epoch_cycles * nb_rollout_steps * nb_epochs)

        # Exploration fraction
        self.explore = 0.35

        for epoch in range(nb_epochs):
            log.info(f"Epoch: {epoch} / {nb_epochs}")
            state = env.reset()

            ############
            # Training #
            ############
            self.train_episode_reward = 0
            for cycle in range(nb_epoch_cycles):
                # log.info(f"Start cycle {cycle}/{nb_epoch_cycles}")
                for t_rollout in range(nb_rollout_steps):
                    # Explore 35% of all steps
                    if (cycle * t_rollout * epoch) < self.number_it * self.explore:
                        # Explore
                        action = env.action_space.sample()
                    else:
                        # Predict next action.
                        action = self.ddpg_agent.get_action(state)
                        action += gaussian_noise()

                        action = np.clip(action,
                                         env.action_space.low,
                                         env.action_space.high)

                    new_state, reward, done, _ = env.step(action)

                    t += 1

                    self.episode_reward += reward
                    self.train_episode_reward += reward
                    self.episode_step += 1

                    # Book-keeping.
                    self.epoch_actions.append(action)

                    # Gather experiences
                    replay_buffer.add(state, action, new_state, reward, done)

                    state = new_state

                    if done:
                        # Episode done.
                        self.epoch_episode_rewards.append(self.episode_reward)
                        self.episode_rewards_history.append(self.episode_reward)
                        self.epoch_episode_steps.append(self.episode_step)
                        self.epoch_episodes += 1
                        self.episodes += 1

                # Important: Set an appropriate replay buffer.
                self.ddpg_agent.memory_buffer = replay_buffer

                ################
                # Update Agent #
                ################
                # log.info(f"Start Training ({nb_train_steps})")
                for t_train in range(nb_train_steps):
                    self.ddpg_agent.update(batch_size)

            #####################
            # Trainings rewards #
            #####################
            self.train_episode_reward /= (nb_epoch_cycles * nb_rollout_steps)
            self.train_rewards.append(self.train_episode_reward)

            ########################
            # Evaluation per epoch #
            ########################
            self.eval_episode_reward = 0  # np.zeros(1, dtype=np.float32)
            for t_rollout in range(nb_eval_steps):
                state = env.reset()
                done = False
                while not done:
                    action = self.ddpg_agent.get_action(state)
                    action += gaussian_noise()
                    action = np.clip(action,
                                     env.action_space.low,
                                     env.action_space.high)

                    new_state, reward, done, _ = env.step(action)

                    if False and render:
                        env.render()
                    self.eval_episode_reward += reward

            ######################
            # Evaluation rewards #
            ######################
            self.eval_episode_reward /= nb_eval_steps
            self.eval_rewards.append(self.eval_episode_reward)

            #################
            # Track Results #
            #################
            folder = Path(f'models/{datetime.now().date()}/{name}/')
            folder.mkdir(parents=True, exist_ok=True)

            # Evaluation rewards
            pd.DataFrame(self.eval_rewards).to_csv(f'models/{datetime.now().date()}/{name}/eval_rewards.csv')

            # Training rewards
            pd.DataFrame(self.train_rewards).to_csv(f'models/{datetime.now().date()}/{name}/train_rewards.csv')

            # Config
            cfg = {
                "epochs": nb_epochs,
                "nb_epoch_cycles": nb_epoch_cycles,
                "nb_rollout_steps": nb_rollout_steps,
                "nb_train_steps": nb_train_steps,
                "nb_eval_steps": nb_eval_steps,
                "finished_episodes_train": self.epoch_episodes,
                "mean_reward_training": np.asarray(self.train_rewards).mean(),
                "mean_reward_eval": np.asarray(self.eval_rewards).mean(),
                "exploration_abs": self.number_it * self.explore,
                "exploration": self.explore
            }
            with open(f'models/{datetime.now().date()}/{name}/config.json', "w+") as f:
                json.dump(cfg, f)

            # Agent
            with open(f'models/{datetime.now().date()}/{name}/ddpg_agent_baseline_training.pickle', "wb+") as f:
                pickle.dump(self.ddpg_agent, f)

    def start_training(self, env: object, trials: int = 1,
                       render: bool = False, name: str = None,
                       training_steps: int = None, default: bool = False):
        """ Method to start HPO.

            Parameters:
            -----------
                env: UnityEnvironment or GymEnvironment
                    The environment the agent interacts in.
                trials: int
                    Number of HPO runs that are executed from the HPO
                    library (trials = 2 means the train method is two times
                    executed with different parameters)
                render: bool
                    Flag to decide if we want to render in case of a gym
                    environment.
                name: str
                    A name for the model/agent that is used to store the best
                    model.
                trainings_steps: int
                    Custom number of steps that should be used for training.
                default: bool
                    For deciding to use default parameter (fixed paramter).
        """
        self.default = default

        log.info(f"Optuna set up - trials: {trials}, name: {name}")

        ######################
        # Set Up HPO library #
        ######################
        study = optuna.create_study(direction="maximize")
        log.info("Start optimization!")

        ###########
        # Run HPO #
        ###########
        study.optimize(lambda trial: self.train_hpo(trial, env, render, name,
                                                    study, training_steps),
                       n_trials=trials, timeout=600)

        log.info("Optimization done.")

        ##########################
        # Store study of HPO run #
        ##########################
        with open(f'{self.path}/{name}_study.pickle', 'wb+') as fout:
            pickle.dump(study, fout)

        log.info(f"Study stored. ({self.path}/{name}_study.pickle)")

        env.close()

        return study

    def train_hpo(self, trial: object, env: object, render: bool = False,
                  name: str = None, study: object = None,
                  training_steps: int = 1000):
        """ Trainingsprocedure for running HPO.

            The actual trainingsprocedure is equal to the on in train().
            However here we have additional HPO initilizations.

            Important: This trainingprocedure doesn't have a evaluation loop.

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
                training_steps: int
                    Custom number of training steps that the agent should do.
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
            training_steps = trial.suggest_int("training_steps", 1000, 1500)

        episodes = trial.suggest_int("episodes", training_steps, training_steps * 1.5)
        explore_threshold = trial.suggest_uniform("explore_threshold", 0.15, 0.5)
        batch_size = trial.suggest_int("batch_size", 32, 512)

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

        ###########################
        # Gather env. information #
        ###########################
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        ################
        # Create agent #
        ################
        self.ddpg_agent = DDPGagent(env,
                                    hidden_dim,
                                    actor_lr,
                                    critic_lr,
                                    gamma,
                                    tau)
        self.ddpg_agent.max_action = max_action

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        rewards = []
        noise = OUNoise(env.action_space)

        log.info(f"Start episodes ({episodes}) with {training_steps} steps.")

        self.episode_num = 0

        overall_steps = 0
        for episode in range(episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            if episode % 50 == 0:
                log.info(f"Trainings-Step: {episode}/{episodes}")

            for step in range(training_steps):
                if render:
                    env.render()

                # Explorate the first
                if overall_steps < explore_threshold * (training_steps * episodes):
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps} (Explore)")

                    action = env.action_space.sample()
                else:
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps}")

                    action = self.ddpg_agent.get_action(state)
                    action = noise.get_action(action, step)

                new_state, reward, done, _ = env.step(action)
                # self.ddpg_agent.memory_buffer.push(state, action, reward,
                #                                    new_state, done)
                # Gather experiences
                replay_buffer.add(state, action, new_state, reward, done)

                # Important: Set an appropriate replay buffer.
                self.ddpg_agent.memory_buffer = replay_buffer

                #####################################
                # Train/Update Actor and Critic     #
                # Here we train within the train    #
                # loop. This is a major difference  #
                # to the other trainings procedures #
                #####################################
                if (len(self.ddpg_agent.memory_buffer) > batch_size  and
                    overall_steps >= explore_threshold * (training_steps * episodes)):
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
                    self.episode_num += 1

                overall_steps += 1

            self.track_reward(episode_reward, episode)
            trial.report(episode_reward, episode)

            rewards.append(episode_reward)

        log.info("End episode!")

        ########################
        # HPO Pruning handling #
        ########################
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        ####################
        # Save best models #
        ####################
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
