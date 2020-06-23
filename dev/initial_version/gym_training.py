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
        avg_rewards = []

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
            #print('episode ', episode, 'score %.2f' % rewards,'trailing 100 games avg %.3f' % np.mean(avg_rewards[-100:]))
        x = range(episodes)
        y = rewards
        plot.plot(x, y)
        plot.title("Progress")
        plot.xlabel("episode")
        plot.ylabel("undiscounted return")
        plot.show()

    """ def episode(self, env, agent, noise, nr_episode=0):
         state = env.reset()
         noise.reset()
         undiscounted_return = 0
         discount_factor = 0.99
         training_steps = 50
         done = False
         time_step = 0


         env.render()
         # 1. Select action according to policy
         print("state: " + str(state))
         action = agent.get_action(state)
         action = noise.get_action(action, step)
         #action = model.predict(state)
         # 2. Execute selected action
         next_state, reward, done, _ = env.step(action)
         # 3. Integrate new experience into agent
         td_error = agent.update()
         agent.memory_buffer.push(state, action, reward, next_state, done)
         #agent.memory_buffer.push(state, action, reward, next_state, done,td_error)
         #state = next_state
         undiscounted_return += reward
         time_step += 1
         print(nr_episode, ":", undiscounted_return)
         return undiscounted_return


         env.close()
         print(episodes, ":", reward)
         return reward



        def perform_training(self,env):
        print("Starting training...")
        noise = OUNoise(env.action_space)
        training_episodes = 100
        # Agent setup
        agent = DDPGagent(env)
        returns = [self.episode(self, env, agent, i,noise) for i in range(training_episodes)]
        x = range(training_episodes)
        y = returns
        plot.plot(x, y)
        plot.title("Progress")
        plot.xlabel("episode")
        plot.ylabel("undiscounted return")
        plot.show()
        
        """

