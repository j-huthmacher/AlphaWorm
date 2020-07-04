import torch
import numpy as np
import torch
import gym
import argparse
import os
from datetime import datetime

from td3 import OurDDPG
from td3 import DDPG
from td3.TD3 import TD3
from td3.utils import ReplayBuffer, DynamicExperienceReplay


class TD3_Training:
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, policy, env, seed, eval_episodes=10):
        #eval_env = gym.make(env_name)
        eval_env = env
        #eval_env.seed(seed + 100)
        eval_env.action_space.seed(seed + 100)

        avg_reward = 0.
        episode = 0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                action = np.array(action).reshape((1, 9))
                #print(eval_env.action_space)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
            print("Eval Episode:  " + str(episode))
            episode += 1

        avg_reward /= eval_episodes

        print("---------------------------------------")
        #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print(f"{datetime.now()} \t Evaluation over {eval_episodes} episodes: {avg_reward}")
        print("---------------------------------------")
        return avg_reward

    def start_training(self, env, load):
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="AlphaWorm")  # OpenAI gym environment name (not used to start env in AlphaWorm)
        parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--eval_freq", default=5, type=int)  # How often (time steps) we evaluate
        parser.add_argument("--training_episodes", default=1e3, type=int)  # Time steps initial random policy is used
        parser.add_argument("--max_episodes", default=1e6, type=int)  # Max time steps to run environment
        parser.add_argument("--steps_per_episode", default=1e3, type=int) #Max env steps
        parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
        parser.add_argument("--random_policy_ratio", default=1)  # ratio of random episodes 1 = as many random as policy, 2 = double as many policy as random ...
        parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)  # Discount factor
        parser.add_argument("--tau", default=0.005)  # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
        parser.add_argument("--save_model", default=True, action="store_true")  # Save model and optimizer parameters
        if load:
            parser.add_argument("--load_model", default="default")  # Model load file name, "" doesn't load, "default" uses file_name
        else:
            parser.add_argument("--load_model",
                                default="")  # Model load file name, "" doesn't load, "default" uses file_name
        parser.add_argument("--load_replays", default="")  # Loads pre-trained replays to replay into the buffer "" doesn't load, "..." loads from the specified folder name
        parser.add_argument("--random_policy", default=False) #Activate random policy

        args = parser.parse_args()

        file_name = f"{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"{datetime.now()} \t Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if args.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        if not os.path.exists("./buffers"):
            os.makedirs("./buffers")

        # Set seeds
        #env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
        }

        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3(**kwargs)

        if args.load_model != "":
            policy_file = file_name if args.load_model == "default" else args.load_model
            policy.load(f"./models/{policy_file}")

        replay_buffer = ReplayBuffer(state_dim, action_dim)
        best_buffer = ReplayBuffer(state_dim, action_dim)
        der_buffer = DynamicExperienceReplay(state_dim, action_dim)

        if args.load_replays != "":
            batch = der_buffer.load(args.load_replays, True)
            if batch is not None:
                policy.train(batch, args.batch_size)
            else:
                print("No buffer batch loaded")

        # Evaluate untrained policy
        #evaluations = [self.eval_policy(policy, env, args.seed)]
        evaluations = []
        state, done = env.reset(), False
        episode_reward = 0

        #for t in range(int(args.max_timesteps)):

        for episode in range(int(args.max_episodes)):

            for steps in range(int(args.steps_per_episode)):

                if args.random_policy:
                    # Select action randomly or according to policy
                    if episode % ((args.random_policy_ratio + 1) * args.training_episodes) < args.training_episodes:
                        action = env.action_space.sample()
                    else:
                        action = (
                                policy.select_action(np.array(state))
                                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)
                else:
                    if episode < args.training_episodes:
                        action = env.action_space.sample()
                    else:
                        action = (
                                policy.select_action(np.array(state))
                                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)

                # Perform action
                action = np.array(action).reshape((1, 9))
                next_state, reward, done, _ = env.step(action)
                done = True if steps < args.steps_per_episode else False
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

            print(f"{datetime.now()} \t Episode Num: {episode + 1} Total T: {(episode + 1) * int(args.steps_per_episode)}  Reward: {episode_reward}")

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0

            # Train agent after collecting sufficient data
            if args.random_policy:
                if episode % ((args.random_policy_ratio + 1) * args.training_episodes) >= args.training_episodes:
                    policy.train(replay_buffer, args.batch_size)
                    replay_buffer = ReplayBuffer(state_dim, action_dim)
            else:
                if episode >= args.training_episodes:
                    policy.train(replay_buffer, args.batch_size)
                    replay_buffer = ReplayBuffer(state_dim, action_dim)

            # Evaluate episode
            if (episode + 1) % args.eval_freq == 0:
                evaluations.append(self.eval_policy(policy, env, args.seed))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: policy.save(f"./models/{file_name}")
                if args.load_replays != "":
                    batch = der_buffer.load(args.load_replays, True)
                    if batch is not None:
                        policy.train(batch, args.batch_size)
                    else:
                        print("No buffer batch loaded")

            if(episode + 1) % (args.eval_freq * 50) == 0:
                der_buffer.save()