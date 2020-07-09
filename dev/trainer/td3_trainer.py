""" Implementation of trainer that trains the TD3 agent.

    This class is responsible for the whole training procedure.
    The functions and the concept is used from the td3/training.py and was
    copied for debugging the DDPG implementation.

    We recommend to use the correct td3 training in td3/training.py

    @author: jhuthmacher
"""
####################
# Default Packages #
####################
import json
from datetime import datetime
import pickle
from pathlib import Path

####################
# ML / RL Packages #
####################
import numpy as np
import pandas as pd
import torch

from trainer import Trainer
from agents import TD3agent, DynamicExperienceReplay, ReplayBuffer
from config.config import log


class TD3Trainer(Trainer):
    """ Trainer class to train a DDPG agent.
    """

    def __init__(self):
        """ Initilzation
        """
        super(TD3Trainer, self).__init__()

        ####################
        # Trainings Config #
        ####################
        self.config = {
            "seed": 0,  # Sets Gym, PyTorch and Numpy seeds
            "eval_freq": 5,  # How often (time steps) we evaluate
            "training_episodes": 1e3,  # Time steps initial random policy is used            
            "expl_noise": 0.1,  # Std of Gaussian exploration noise
            "random_policy_ratio": 1,  # ratio of random episodes 1 = as many random as policy, 2 = double as many policy as random ...
            "batch_size": 256,  # Batch size for both actor and critic
            "discount": 0.99,  # Discount factor
            "tau": 0.005,  # Target network update rate
            "policy_noise": 0.2,  # Noise added to target policy during critic update
            "noise_clip": 0.5,  # Range to clip target policy noise
            "policy_freq": 2,  # Frequency of delayed policy updates
            "save_model": True,  # Save model and optimizer parameters
            "episodes": 1e6,
            "training_steps": 1e3,  # Max env steps
            "evaluation_steps": 10,
            "evaluation_lim": None
        }

    def train(self, env: object, render: bool = False, name: str = None):
        """ Train method from td3/training.py

            Parameters:
            -----------
                env: GymEnvironment or UnityEnvironment
                    Environment in which the agent interacts
                rende: bool
                    Falg to decide if the intermediate steps should be rendered
                name: str
                    Name of the model/agent for tracking
        """
        #############
        # Set seeds #
        #############
        env.action_space.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        replay_buffer = ReplayBuffer(state_dim, action_dim)
        best_buffer = ReplayBuffer(state_dim, action_dim)
        der_buffer = DynamicExperienceReplay(state_dim, action_dim)

        # Evaluate untrained policy
        state, done = env.reset(), False
        episode_reward = 0

        # Init directory set up.
        self.track_setup(name)

        ################
        # Create agent #
        ################
        self.td3_agent = TD3agent(state_dim, action_dim,
                                  self.config["discount"],
                                  self.config["tau"],
                                  self.config["policy_noise"],
                                  self.config["noise_clip"],
                                  self.config["policy_freq"])

        rewards = []

        episodes = self.config["episodes"]
        training_steps = self.config["training_steps"]

        log.info(f"Start episodes ({self.config['episodes']}) with {self.config['training_steps']} steps.")

        self.episode_num = 0
        self.eval_rewards = []
        self.train_rewards = []

        log.info("Parameter:")
        log.info(self.config)

        #############################
        # Start Trainings Procedure #
        #############################
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            if episode % 50 == 0:
                log.info(f"Episode-Step: {episode}/{episodes}")

            ############
            # Training #
            ############
            for step in range(training_steps):

                ###############
                # Exploration #
                ###############
                if episode < self.config["training_episodes"]:
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps} (Explore)")

                    action = env.action_space.sample()
                else:
                    if step % 50 == 0:
                        log.info(f"Trainings-Step: {step}/{training_steps}")
                    action = self.td3_agent.select_action(np.array(state))
                    noise = np.random.normal(0, max_action * self.config["expl_noise"], size=action_dim)

                    action = (action + noise).clip(-max_action,
                                                   max_action)

                # Important for the Pendulum domain.
                if np.array(action).size > 1:
                    action = np.array(action).reshape((1, 9))

                next_state, reward, done, _ = env.step(action)
                done = (True
                        if step < self.config["training_steps"]
                        else False)
                done_bool = float(done)

                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done_bool)
                best_buffer.add(state, action, next_state, reward, done_bool)

                # Store buffer
                if done:
                    der_buffer.add(best_buffer)
                    best_buffer = ReplayBuffer(state_dim, action_dim)

                state = next_state
                episode_reward += reward

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0

            #################################
            # Update neural nets (Learning) #
            #################################
            if episode >= self.config["training_episodes"]:
                # Only train after exploration
                self.td3_agent.train(replay_buffer, self.config["batch_size"])
                # Reset replay buffer after training
                # replay_buffer = ReplayBuffer(state_dim, action_dim)

            ########################
            # Evaluation per epoch #
            ########################
            log.info(f"Start Evaluation: {self.config['evaluation_steps']}")
            eval_env = env
            eval_env.action_space.seed(self.config['seed'] + 100)

            avg_reward = 0.
            episode = 0
            k = 0
            for _ in range(self.config['evaluation_steps']):
                state, done = eval_env.reset(), False
                while not done:
                    action = self.td3_agent.select_action(np.array(state))
                    action = np.array(action).reshape((1, 9))
                    # print(eval_env.action_space)
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

                    if self.config["evaluation_lim"] is not None and self.config["evaluation_lim"] < k:
                        break

                    k += 1

                log.info("Eval Episode:  " + str(episode))
                episode += 1

            avg_reward /= self.config['evaluation_steps']

            log.info(f"Evaluation Reward: {avg_reward}")

            self.eval_rewards.append(avg_reward)

            self.track_reward(episode_reward, episode)
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
            with open(f'models/{datetime.now().date()}/{name}/td3_agent_trained.pickle', "wb+") as f:
                pickle.dump(self.td3_agent, f)

        log.info("End episode!")
