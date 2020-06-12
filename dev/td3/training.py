import torch

from td3.ReplayBuffer import ReplayBuffer
from td3.Runner import Runner

from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import sys

ENV = "RoboschoolHalfCheetah-v1"#"Pendulum-v0"
SEED = 0
OBSERVATION = 10000
EXPLORATION = 5000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000


class TD3_Training:
    def start_training(env):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds
        env.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        policy = TD3(state_dim, action_dim, max_action, env, device)

        replay_buffer = ReplayBuffer()

        runner = Runner(env, policy, replay_buffer)

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        observe(env, replay_buffer, OBSERVATION)

        train(policy, env, runner, replay_buffer)

        policy.load()

        for i in range(100):
            evaluate_policy(policy, env, render=True)

        env.close()



def evaluate_policy(policy, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """

    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def observe(env, replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for

    """

    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


def train(agent, env, runner, replay_buffer):
    """Train the agent for exploration steps

        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run

    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    obs = env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000

    writer = SummaryWriter(comment="-TD3_Baseline_HalfCheetah")

    while total_timesteps < EXPLORATION:

        if done:

            if total_timesteps != 0:
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg", "saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()

                if avg_reward >= REWARD_THRESH:
                    break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP,
                            POLICY_FREQUENCY)

                # Evaluate episode
                #                 if timesteps_since_eval >= EVAL_FREQUENCY:
                #                     timesteps_since_eval %= EVAL_FREQUENCY
                #                     eval_reward = evaluate_policy(agent, test_env)
                #                     evaluations.append(avg_reward)
                #                     writer.add_scalar("eval_reward", eval_reward, total_timesteps)

                #                     if best_avg < eval_reward:
                #                         best_avg = eval_reward
                #                         print("saving best model....\n")
                #                         agent.save("best_avg","saves")

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        reward, done = runner.next_step(episode_timesteps)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1