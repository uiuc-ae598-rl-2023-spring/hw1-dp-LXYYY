import numpy as np
from models.policy_iteration.policy_iteration_learner import PolicyIteration


def learn(env, scene, max_it, load=None, train=True, **kwargs):
    agent = PolicyIteration(env, scene=scene, gamma=0.95)
    if load is not None:
        agent.load_checkpoint(load)
    if train:
        for epoch in range(max_it):
            print(f'Epoch {epoch}')

            env.reset()
            agent.policy_eval()

            env.reset()
            stable = agent.policy_improvement()

            print('Policy:')
            print(np.round(agent.get_policy().reshape(5, 5), 1))

            print('Values:')
            print(np.round(agent.get_values().reshape(5, 5), 1))

            if stable:
                break

    return agent
