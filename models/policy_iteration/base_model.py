import abc
import numpy as np


# define an abc models class
class BaseModel(abc.ABC):
    def __init__(self, env):
        self.env = env
        self._values = np.random.rand(int(np.sqrt(self.env.num_states)), int(np.sqrt(self.env.num_states)))
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(0, self.env.num_actions, self._values.shape)

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
