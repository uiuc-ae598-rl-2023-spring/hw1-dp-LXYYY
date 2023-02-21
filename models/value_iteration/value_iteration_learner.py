from models.policy_iteration.policy_iteration_learner import PolicyIteration
import numpy as np


class ValueIteration(PolicyIteration):
    def __init__(self, env, gamma=0.95, theta=1e-9, max_iterations=1000):
        super().__init__(env, gamma, theta, max_iterations)
        self.log.experiment='value_iteration'

    def value_iteration(self):
        i = 0
        delta = 0
        for s in range(self.env.num_states):
            max_a = -1
            max_v = -np.inf
            old_v = self.get_values(s)
            for a in range(self.env.num_actions):
                _, new_value=self.eval_state(s, a)
                if new_value > max_v:
                    max_v = new_value
                    max_a = a
            self.set_values(s, max_v)
            self.set_policy(s, max_a)
            delta = max(delta, abs(old_v - max_v))

        print(f'delta: {delta}')

        self.log.add('mean_value', self.get_mean_value())

        if delta < self.theta:
            return True, delta
        else:
            return False, delta
