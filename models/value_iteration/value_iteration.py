import numpy as np
from models.value_iteration.value_iteration_learner import ValueIteration


def learn(env, max_it, theta=1e-09, **kwargs):
    agent = ValueIteration(env,theta=theta)

    epoch = 0
    reached_theta = False
    for epoch in range(max_it):
        reached_theta, delta = agent.value_iteration()
        print('Policy:')
        print(agent.get_policy_mat())
        print('Values:')
        print(agent.get_values_mat())
        if reached_theta:
            break

    print('Learning finished after {} epochs, converged {}, delta {}'.format(epoch, reached_theta, delta))

    return agent
