import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import pickle

import os

class InitialTrainingExample:
    def start_training(env):



        log_dir = "pp02_results"
        os.makedirs(log_dir, exist_ok=True)

        # Continue learning already learned model
        loaded_model = PPO2.load(log_dir + "/model")
        loaded_model.set_env(env)
        #loaded_model.learn(total_timesteps=10000)
        model = loaded_model
        #model = PPO2(MlpPolicy, env, verbose=1)
        #model.learn(total_timesteps=10000)

        #evaluate agent
        episodes = 1000
        ep_r = []
        ep_l = []
        for e in range(episodes):
            obs = env.reset()
            total_r = 0.
            total_l = 0.
            while True:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_l += 1.
                total_r += reward
                if done:
                    break
            ep_r.append(total_r)
            ep_l.append(total_l)
        print("episode mean reward: {:0.3f} mean length: {:0.3f}".format(np.mean(ep_r), np.mean(ep_l)))
        with open('{}/eval.pkl'.format(log_dir), 'wb') as f:
            pickle.dump(ep_r, f)
            pickle.dump(ep_l, f)

        model.save(log_dir + "/model")


