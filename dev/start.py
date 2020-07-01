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

from utils.mlagent_utils import get_env
from trainer.ddpg_trainer import DDPGTrainer
from config.config import log


#PATH TO ALGORITHM
from initial_version.training import InitialTrainingExample

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
            env = UnityToGymWrapper(unity_env, rank, uint8_visual=False)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])


def main():
    #   Set to FALSE for CIP-Pool execution
    # env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)
    # InitialTrainingExample.start_training(env)
    # env.close()

    train_ddpg()


def train_ddpg():
    """
        @author: jhuthmacher
    """
    # env = "envs/worm_dynamic_one_agent/win/UnityEnvironment"
    env = "./envs/worm_dynamic_one_agent/linux/worm_dynamic"
    env = get_env(env, False)

    trainer = DDPGTrainer()

    log.info("Start DDPG training (WormDomain)...")

    # study = trainer.start_training(env, trials=2, render=False,
    #                                name="WormDomain-5_Default",
    #                                default=True)

    study = trainer.start_training(env, trials=1, render=False, default=True,
                                   name="DPPG-WormDomain-8-FixedParameter")

    log.info("Training done!")


if __name__ == '__main__':
    main()
