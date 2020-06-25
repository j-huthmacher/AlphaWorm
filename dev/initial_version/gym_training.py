import gym
import matplotlib.pyplot as plot
from dev.models.ddpg.ddpg import DDPGagent
import numpy as np
from dev.models.ddpg.OUNoise import OUNoise

class GymTraining:


    def perform_training(self, env):
        print("Starting training...")
        noise = OUNoise(env.action_space)
        # Agent setup
        agent = DDPGagent(env)
        episodes = 50
        training_steps = 50

        batch_size = 128
        rewards = []

        for episode in range(episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            for step in range(training_steps):
                action = agent.get_action(state)
                action = noise.get_action(action, step)
                new_state, reward, done, _ = env.step(action)
                # Unsqueeze new state to prevent input mismatch
                new_state = np.squeeze(new_state)
                agent.memory_buffer.push(state, action, reward, new_state, done)

                #if len(agent.memory_buffer) > batch_size:
                td_error = agent.update(batch_size)

                #Update placeholder in buffer
                if td_error is not None:
                    agent.memory_buffer.priorities.pop()
                    agent.memory_buffer.priorities.append(td_error)



                state = new_state
                episode_reward += reward[0]
            env.close()
            print(episode, ":", episode_reward)
            rewards.append(episode_reward)
        x = range(episodes)
        y = rewards
        plot.plot(x, y)
        plot.title("Progress")
        plot.xlabel("episode")
        plot.ylabel("undiscounted return")
        plot.show()
