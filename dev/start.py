import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import logger
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import pickle
import os
import time
import gym

from utils.mlagent_utils import get_env
from trainer.ddpg_trainer import DDPGTrainer
from config.config import log
import gym


#PATH TO ALGORITHM
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
    #start_unity()
    start_unity_baselines()
    #start_gym_std()


def train_ddpg():
    """
        @author: jhuthmacher
    """
    # env = "envs/worm_dynamic_one_agent/win/UnityEnvironment"
    env = "./envs/worm_dynamic_one_agent/linux/worm_dynamic"
    env = get_env(env, False)

    trainer = DDPGTrainer()

    log.info("Start DDPG training (WormDomain)...")

    # trainer.train(env, name="DPPG-WormDomain-9-GaussianNoise-Clipping")

    # trainer.train_baseline(env, name="DPPG-WormDomain-10-Baseline",
    #                        nb_epochs=1000, nb_epoch_cycles=50, nb_rollout_steps=200,
    #                        nb_train_steps=500, nb_eval_steps=500)
    # trainer.config["episodes"] = 1500
    # trainer.config["training_steps"] = 10
    # trainer.config["episodes"] = 10
    # trainer.config["training_steps"] = 10
    # trainer.config["evaluation_lim"] = 10
    trainer.train(env, name="DPPG-WormDomain")

    log.info("Training done!")


def train_ddpg_gym(env_name: str = "Pendulum-v0"):
    """
    """
    env = gym.make(env_name)

    trainer = DDPGTrainer()

    log.info(f"Start DDPG training ({env_name})...")

    trainer.config["episodes"] = 1000
    trainer.config["training_steps"] = 700

    trainer.train(env, name="DPPG-{env_name}")

    # trainer.train_baseline(env, name=f"DPPG-{env_name}-2-Longer-Training",
    #                        nb_epochs=300, nb_epoch_cycles=100, nb_rollout_steps=300,
    #                        nb_train_steps=600, nb_eval_steps=600)

    # trainer.train_baseline(env, name=f"DPPG-{env_name}-2", render=False)

    log.info("Training done!")
    
def start_unity_baselines():
    #   Set to FALSE for CIP-Pool execution
    # env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)
    # InitialTrainingExample.start_training(env)
    # env.close()

    train_ddpg()
    # train_ddpg_gym()
    env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3_Baselines(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=int(2e6), log_interval=10)
    model.save("td3_worm")

def start_unity():
    #   Set to FALSE for CIP-Pool execution
    env = make_unity_env('./envs/worm_dynamic_one_agent/linux/worm_dynamic', 1, False)

    training = TD3_Training()
    training.start_training(env, load=False, der_activated=False)
    env.close()

def start_gym_std():
    #env = gym.make("MountainCarContinuous-v0")
    env = gym.make("Pendulum-v0")

    #Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, render=False, load=False, der_activated=False)
    env.close()


if __name__ == '__main__':
    main()
