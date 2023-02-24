import numpy as np
from models.value_iteration.value_iteration_learner import ValueIteration


def learn(env, scene, max_it, theta=1e-09, load=None, **kwargs):
    agent = ValueIteration(env, scene=scene, theta=theta)

    if load is not None:
        agent.load_checkpoint(load)
    else:
        epoch = 0
        reached_theta = False
        delta = 0
        for epoch in range(max_it):
            reached_theta, delta = agent.value_iteration()
            print('Policy:')
            print(agent.get_policy())
            print('Values:')
            print(agent.get_values())
            if reached_theta:
                break

        print('Learning finished after {} epochs, converged {}, delta {}'.format(epoch, reached_theta, delta))

    return agent
