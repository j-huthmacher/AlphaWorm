""" Start script for execute the implemented algorithm using the console.

    Starting point is the 'dev' directiory!
"""

import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import logger
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import os
from pathlib import Path
from datetime import datetime

from config.config import log, logFormatter

from utils.mlagent_utils import get_env
from trainer.ddpg_trainer import DDPGTrainer
from config.config import log


# PATH TO ALGORITHM
from stable_baselines.td3 import MlpPolicy
from stable_baselines import TD3 as TD3_Baselines


from td3.training import TD3_Training
from td3.training_gym import TD3_Training_Gym

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True): # pylint: disable=C0111
        def _thunk():
            no_graphics = not use_visual
            unity_env = UnityEnvironment(env_directory, no_graphics=no_graphics)
            env = UnityToGymWrapper(unity_env, uint8_visual=False)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])


def main():
    # Start TD3 Unity Training
    start_unity()

    # Start TD3 Baseline Unity Training
    # start_unity_baselines()

    # Start TD3 Gym Training
    # start_gym_std()

    # Start DDPG Unity Training
    # train_ddpg()

    # Start DDPG Gym Training
    # train_ddpg_gym()


def train_ddpg(name: str = "DPPG-WormDomain"):
    """ Train the DDPG on the unity ML environment.
    """
    ###########################################
    # Ensure that the path exists for logging #
    ###########################################
    folder = Path(f'models/{datetime.now().date()}/{name}/')
    folder.mkdir(parents=True, exist_ok=True)

    # Store logs directly nearby the results!
    fh = log.FileHandler(f'models/{datetime.now().date()}/{name}/{datetime.now().date()}.log')
    fh.setFormatter(logFormatter)
    log.getLogger().addHandler(fh)
    
    env = "envs/worm_dynamic_one_agent/win/UnityEnvironment"  # Windows
    # env = "./envs/worm_dynamic_one_agent/linux/worm_dynamic"  # Linux
    env = get_env(env, False)

    trainer = DDPGTrainer()

    log.info("Start DDPG training (WormDomain)...")

    trainer.train(env, name=name)

    log.info("Training done!")


def train_ddpg_gym(env_name: str = "Pendulum-v0"):
    """ Train the DDPG on a gym environment
    """
    ###########################################
    # Ensure that the path exists for logging #
    ###########################################
    folder = Path(f'models/{datetime.now().date()}/DPPG-{env_name}/')
    folder.mkdir(parents=True, exist_ok=True)

    # Store logs directly nearby the results!
    fh = log.FileHandler(f'models/{datetime.now().date()}/DPPG-{env_name}/{datetime.now().date()}.log')
    fh.setFormatter(logFormatter)
    log.getLogger().addHandler(fh)


    env = gym.make(env_name)

    trainer = DDPGTrainer()

    log.info(f"Start DDPG training ({env_name})...")

    # Define default parameter
    # trainer.config["episodes"] = 1000
    # trainer.config["training_steps"] = 700

    trainer.train(env, name=f"DPPG-{env_name}")
    log.info("Training done!")


def start_unity_baselines():
    # Set to FALSE for CIP-Pool execution
    # env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)
    # InitialTrainingExample.start_training(env)
    # env.close()

    unity_env = UnityEnvironment('./envs/worm_dynamic_one_agent/linux/worm_dynamic', no_graphics=True)
    env = UnityToGymWrapper(unity_env, uint8_visual=False)
    env = Monitor(env, 'results/')
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3_Baselines(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=int(2e6), log_interval=10)
    model.save("td3_worm")


def start_unity():
    # Set to FALSE for CIP-Pool execution
    env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)

    training = TD3_Training()
    training.start_training(env, load=False, der_activated=False)
    env.close()


def start_gym_std():
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("Pendulum-v0")

    # Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, render=False, load=False, der_activated=False)
    env.close()


if __name__ == '__main__':
    main()
