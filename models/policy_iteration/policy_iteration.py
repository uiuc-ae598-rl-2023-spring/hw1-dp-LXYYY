import numpy as np
from models.policy_iteration.policy_iteration_learner import PolicyIteration


def learn(env, **kwargs):
    agent = PolicyIteration(env, env.num_states, env.num_actions, gamma=0.9)

    agent.policy_eval()

    print(np.round(agent.values.reshape([int(np.sqrt(env.num_states)),
                                         int(np.sqrt(env.num_states))]),
                   1))
