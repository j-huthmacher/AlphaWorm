"""
    @author: jhuthmacher
"""
import optuna
from optuna.trial import FixedTrial
import pickle

import numpy as np
from trainer.trainer import Trainer
from agents import TD3agent, OUNoise, GaussianNoise, DynamicExperienceReplay, ReplayBuffer
from config.config import log
from shutil import copyfile
import json
from collections import deque
import os
from datetime import datetime
import pandas as pd
import pickle
from pathlib import Path
import torch


class TD3Trainer(Trainer):
    """ Trainer class to train a DDPG agent.
    """

    def __init__(self):
        """ Initilzation
        """
        super(TD3Trainer, self).__init__()

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
            "training_steps": 1e3, # Max env steps
            "evaluation_steps": 10,
            "evaluation_lim": None
        }

    def train(self, env: object, render: bool = False, name: str = None):
        """
        """
        # Set seeds
        # Set seeds
        # env.seed(args.seed)
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
        noise = OUNoise(env.action_space)

        episodes = self.config["episodes"]
        training_steps = self.config["training_steps"]

        log.info(f"Start episodes ({self.config['episodes']}) with {self.config['training_steps']} steps.")

        self.episode_num = 0
        self.eval_rewards = []
        self.train_rewards = []

        log.info("Parameter:")
        log.info(self.config)

        for episode in range(episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            if episode % 100 == 0 or (episode < 10):
                log.info(f"Episode-Step: {episode}/{episodes}")

            ############
            # Training #
            ############
            for step in range(training_steps):
                if render:
                    env.render()

                ###############
                # Exploration #
                ###############
                if episode < self.config["training_episodes"]:
                    if step % 100 == 0 or (step < 10):
                        log.info(f"Trainings-Step: {step}/{training_steps} (Explore)")

                    action = env.action_space.sample()
                else:
                    if step % 100 == 0 or (step < 10):
                        log.info(f"Trainings-Step: {step}/{training_steps}")

                    action = (
                                self.td3_agent.select_action(np.array(state))
                                + np.random.normal(0, max_action * self.config["expl_noise"], size=action_dim)
                        ).clip(-max_action, max_action)

                log.info(action.shape)

                # Perform action
                action = np.array(action).reshape((1, 9))
                
                next_state, reward, done, _ = env.step(action)
                done = (True
                        if step < self.config["training_steps"]
                        else False)
                done_bool = float(done)

                log.info(action.shape)

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

            if episode >= self.config["training_episodes"]:
                # Only train after exploration
                self.td3_agent.train(replay_buffer, self.config["batch_size"])
                replay_buffer = ReplayBuffer(state_dim, action_dim)

            ########################
            # Evaluation per epoch #
            ########################
            log.info(f"Start Evaluation: {self.config['evaluation_steps']}")
            # eval_env = gym.make(env_name)
            eval_env = env
            # eval_env.seed(seed + 100)
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

                    if self.config["evaluation_lim"] != None and self.config["evaluation_lim"] < k:
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
            pd.DataFrame(self.eval_rewards).to_csv(f'models/{datetime.now().date()}/{name}/eval_rewards.csv')

            # Training rewards
            # pd.DataFrame(self.train_rewards).to_csv(f'models/{datetime.now().date()}/{name}/train_rewards.csv')

            with open(f'models/{datetime.now().date()}/{name}/config.json', "w+") as f:
                json.dump(self.config, f)

            # Agent
            with open(f'models/{datetime.now().date()}/{name}/td3_agent_trained.pickle', "wb+") as f:
                pickle.dump(self.td3_agent, f)

        log.info("End episode!")
