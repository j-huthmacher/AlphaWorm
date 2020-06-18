import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import logger
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import pickle
import os
import time
import gym

#PATH TO ALGORITHM
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
    start_unity()
    #start_gym_std()

def start_unity():
    #   Set to FALSE for CIP-Pool execution
    env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, True)

    training = TD3_Training()
    training.start_training(env, True)
    env.close()

def start_gym_std():
    #env = gym.make("MountainCarContinuous-v0")
    env = gym.make("Pendulum-v0")

    #Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, False, False)
    env.close()


if __name__ == '__main__':
    main()
