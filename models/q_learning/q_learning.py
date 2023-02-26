from models.q_learning.q_learning_learner import QLearning
import numpy as np


def learn(env, scene, max_it, epsilon, alpha, **kwargs):
    agent = QLearning(env, scene=scene, epsilon=epsilon, alpha=alpha)

    for episode in range(max_it):
        # Initialize S
        env.reset()
        done = False

        return_per_episode = 0
        while not done:
            s = env.s
            # Choose A from S using episilon-greedy policy
            a = agent.get_a(env.s, agent.epsilon)
            # Take A, observe R, S'
            s_, r, done = env.step(a)
            agent.update_Q(s, a, r, s_, None, done)

            return_per_episode += r

        # print(agent.Q)
        # print(np.linalg.norm(agent.Q-old_q))
        agent.plot.add('return_per_episode', return_per_episode, xlabel='episode', ylabel='return',
                       title='Return per Episode of ' + agent.algorithm + ' in ' + agent.scene)

        if episode % (0.1 * max_it) == 0:
            print(f'Episode {episode} of {max_it} finished')


    print(agent.Q)

    print(agent.get_policy())

    return agent
