import numpy as np

class Runner():
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False

    def next_step(self, episode_timesteps, noise=0.1):
        action = self.agent.select_action(np.array(self.obs), noise=0.1)
        action = np.array(action).reshape((1, 9))
        # Perform action
        new_obs, reward, done, _ = self.env.step(action)
        done_bool = 0 if episode_timesteps + 1 == 200 else float(done)

        # Store data in replay buffer
        self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

        self.obs = new_obs

        if done:
            self.obs = self.env.reset()
            done = False

            return reward, True

        return reward, done