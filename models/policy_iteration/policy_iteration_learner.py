import numpy as np


class PolicyIteration(object):
    def __init__(self, env, s_shape, a_shape, gamma=0.95, theta=1e-8, max_iter_eval=1000):
        self.values = np.random.rand(s_shape) * 20
        # generate random policy with uniform distribution sum to 1
        self.policy = np.random.rand(a_shape, s_shape)
        self.policy = self.policy / np.sum(self.policy, axis=0)
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iter_eval = max_iter_eval

    def store_transition(self, s, a, r, s_, done):
        pass

    def policy(self, s):
        pass

    def step(self):
        # take max action
        s = self.env.s
        a = np.argmax(self.policy[:, s])
        # env step
        s_, r, done = self.env.step(a)
        if not done:
            assert self.env.p(s_, s, a) == 1, 'p(s_, s, a) should not be 1'
            # calculate the value of the state
            delta = self.step()
            v = self.env.p(s_, s, a) * (r + self.gamma * self.values[s_])
            delta = max(delta, np.abs(v - self.values[s]))
            self.values[s] = v
            return delta
        return 0

    def policy_eval(self):
        self.env.reset()
        i = 0
        delta = 0
        for i in range(self.max_iter_eval):
            # iterate all states
            self.env.reset()
            delta = self.step()

            # calculate the difference between new value and old value
            # if the difference is smaller than theta, stop the iteration
            if delta < self.theta:
                break

        print(f'policy evaluation finished after {i + 1} iterations')
        print(delta)

    def policy_improvement(self):
        pass
