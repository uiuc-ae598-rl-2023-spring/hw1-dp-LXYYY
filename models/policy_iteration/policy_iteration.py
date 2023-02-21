import numpy as np
from models.policy_iteration.policy_iteration_learner import PolicyIteration


def learn(env, max_it, **kwargs):
    agent = PolicyIteration(env, gamma=0.95)

    for epoch in range(max_it):
        print(f'Epoch {epoch}')

        env.reset()
        agent.policy_eval()

        env.reset()
        stable = agent.policy_improvement()

        print('Policy:')
        print(np.round(agent.get_policy_mat(), 1))

        print('Values:')
        print(np.round(agent.get_values_mat(), 1))

        if stable:
            break

    return agent
