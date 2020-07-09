"""
    This script contains helper and utility function to
    simplify some tasks.

    @author: jhuthmacher
"""

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def get_env(env_directory: str, visual: bool, start_index: int = 0):
    """ For loading the evironment
    """

    unity_env = UnityEnvironment(env_directory, seed=0,
                                 no_graphics=not visual)
    env = UnityToGymWrapper(unity_env, uint8_visual=False)

    return env
