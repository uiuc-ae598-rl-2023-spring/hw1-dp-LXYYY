import numpy as np


class PolicyIteration(object):
    def __init__(self, env, gamma=0.95, theta=1e-12, max_iter_eval=100):
        self.env = env
        self._values = np.zeros([int(np.sqrt(self.env.num_states)), int(np.sqrt(self.env.num_states))],
                                dtype=np.float32)
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(0, self.env.num_actions, self._values.shape)
        self.gamma = gamma
        self.theta = theta
        self.max_iter_eval = max_iter_eval

    def get_pos(self, s):
        return self.env.get_pos(s)

    def get_values(self, s):
        return self._values[self.get_pos(s)]

    def set_values(self, s, v):
        self._values[self.get_pos(s)] = v

    def get_values_mat(self):
        return self._values

    def get_policy(self, s):
        return self._policy[self.get_pos(s)]

    def set_policy(self, s, a):
        self._policy[self.get_pos(s)] = a

    def get_policy_mat(self):
        return self._policy

    # Recursive step
    def step(self, policy_eval=False):
        # take max action
        s = self.env.s

        if s == 0:
            t = 1

        max_a = -1
        max_v = -np.inf
        if not policy_eval:
            # iterate all next states
            for s_ in range(self.env.num_states):
                for a in range(self.env.num_actions):
                    if self.env.p(s_, s, a) == 1:
                        if self.get_values(s_) > max_v:
                            max_a = a
                            max_v = self.get_values(s_)

            assert max_a != -1, 'max_a should not be -1'
            self.set_policy(s, max_a)

        a = self.get_policy(s)

        # env step
        s_, r, done = self.env.step(a)
        if not done:
            assert self.env.p(s_, s, a) == 1, 'p(s_, s, a) should be 1'
            # calculate the value of the state
            delta = self.step()
            if policy_eval:
                v = self.env.p(s_, s, a) * (r + self.gamma * self.get_values(s_))
                delta = max(delta, np.abs(v - self.get_values(s)))
                self.set_values(s, v)
                return delta

        return 0

    def policy_eval(self):
        self.env.reset()
        i = 0
        delta = 0
        for i in range(self.max_iter_eval):
            # iterate all states
            self.env.reset()
            delta = self.step(True)

            # calculate the difference between new value and old value
            # if the difference is smaller than theta, stop the iteration
            if delta < self.theta:
                break

        print(f'policy evaluation finished after {i + 1} iterations with delta= {delta}')

    def policy_improvement(self):
        self.env.reset()
        policy_stable = True
        old_policy = self.get_policy_mat().copy()

        self.step(policy_eval=False)

        policy_delta = np.linalg.norm(self.get_policy_mat() - old_policy)

        print(self.get_policy_mat())
        print(old_policy)

        if policy_delta != 0:
            policy_stable = False
        print(f'policy improvement finished with policy delta: {policy_delta}')

        return policy_stable
