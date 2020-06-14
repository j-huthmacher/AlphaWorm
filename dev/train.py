"""
    Training procedure incl. HPO integration.

    @author: jhuthmacher
"""
import optuna
import numpy as np

from models.ddpg.ddpg import DDPGagent
from models.ddpg.OUNoise import OUNoise

from utils.mlagent_utils import get_env


def train_ddpg():
    """
    """

    def train(trial, env):
        """
        """

        ############################
        # Hyperparameter for Agent #
        ############################
        num_hidden_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dim = []

        for i in range(num_hidden_layers):
            hidden_dim += [trial.suggest_int("n_units_l{}".format(i), 4, 128)]

        actor_lr = trial.suggest_uniform("actor_lr", 1e-6, 1e-1)
        critic_lr = trial.suggest_uniform("critic_lr", 1e-6, 1e-1)

        gamma = trial.suggest_uniform("gamma", 0.001, 0.999)
        tau = trial.suggest_uniform("tau", 1e-6, 1e-1)

        ddpg_agent = DDPGagent(env, hidden_dim, actor_lr, critic_lr,
                               gamma, tau)

        ######################################
        # Hyperparameter for Training set up #
        ######################################
        episodes = trial.suggest_int("episodes", 2, 5) # 50 - 2048
        training_steps = trial.suggest_int("training_steps", 10, 15) # 10 - 256
        batch_size = trial.suggest_int("batch_size", 10, 15) # 10 - 256

        rewards = []
        noise = OUNoise(env.action_space)

        for episode in range(episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            for step in range(training_steps):
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

        print("episode done")


        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return np.array(rewards).mean()

    win_env = "../envs/worm_dynamic_one_agent/win/UnityEnvironment"
    env = get_env(win_env, True)

    ##########
    # Optuna #
    ##########
    # objective(FixedTrial({'x': 1.0, 'y': 0}))

    study = optuna.create_study(direction="maximize")
    # Use lambda to handover custom parameters like the env
    study.optimize(lambda trial: train(trial, env), n_trials=2, timeout=600)

    env.close()

    return study
