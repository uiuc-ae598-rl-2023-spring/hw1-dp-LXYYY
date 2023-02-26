import abc
import numpy as np
import os
from models.plot import Plot


# define an abc models class
class BaseModel(abc.ABC):

    def __init__(self, env, scene, algorithm):
        self.env = env
        self._values = np.random.rand(self.env.num_states)
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(
            0, self.env.num_actions, self._values.shape)
        self.plot = Plot(env=env, scene=scene, algorithm=algorithm)
        self.algorithm = algorithm
        self.scene = scene

    @abc.abstractmethod
    def get_algorithm_name(self, args):
        pass

    def get_pos(self, s):
        return s

    def get_values(self, s=None):
        return self._values[self.get_pos(s)] if s is not None else self._values

    def set_values(self, s, v):
        self._values[self.get_pos(s)] = v

    def get_policy(self, s=None):
        return self._policy[self.get_pos(s)] if s is not None else self._policy

    def set_policy(self, s, a):
        self._policy[self.get_pos(s)] = a

    def get_mean_value(self):
        return np.mean(self._values)

    def get_log(self):
        return self.plot

    def save_values(self, path):
        np.save(path, self.get_values())

    def save_policy(self, path):
        np.save(path, self.get_policy())

    def get_state_value_function(self):
        return self.get_values()

    def save_checkpoint(self, path):
        # join path and create directory if not exist
        path = os.path.join(path, self.scene, self.algorithm)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_values(os.path.join(path, 'value.npy'))
        self.save_policy(os.path.join(path, 'policy.npy'))

    def load_checkpoint(self, path):
        # join path and create directory if not exist
        path = os.path.join(path, self.scene, self.algorithm)
        self._values = np.load(os.path.join(path, 'value.npy'))
        self._policy = np.load(os.path.join(path, 'policy.npy'))


class ModelBasedAlg(BaseModel):
    def __init__(self, env, scene, algorithm, gamma, theta, max_it):
        super().__init__(env, scene, algorithm)
        self.gamma = gamma
        self.theta = theta
        self.max_it = max_it

    def eval_state(self, s, a=None):
        # take max action
        v = self.get_values(s)

        # iterate all next states
        if a is None:
            a = self.get_policy(s)
        new_value = 0
        for s_ in range(self.env.num_states):
            new_value += self.env.p(s_, s, a) * (self.env.r(s, a) + self.gamma * self.get_values(s_))

        return np.abs(v - new_value), new_value


class ModelFreeAlg(BaseModel):
    def __init__(self, env, scene, algorithm, alpha, epsilon, gamma=0.95):
        super().__init__(env, scene, algorithm)
        self.Q = np.random.rand(self.env.num_states, self.env.num_actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.policy = 'epsilon_greedy'
        self.value_learning_method = 'TD0'

    @staticmethod
    def get_model_free_alg_name(args):
        epsilon, alpha, alg = args[0], args[1], args[2]
        return alg + r' $\epsilon$=' + str(epsilon) + r' $\alpha$=' + str(alpha)

    @abc.abstractmethod
    def Q_s_(self, s_, a_):
        return .0

    def update_Q(self, s, a, r, s_, a_, done):
        if self.value_learning_method == 'TD0':
            return self.update_Q_TD0(s, a, r, s_, a_, done)
        else:
            raise NotImplementedError

    def update_Q_TD0(self, s, a, r, s_, a_, done):
        q = self.get_Q(s, a)
        Q_s_ = self.Q_s_(s_, a_) if not done else 0
        q += self.alpha * (r + self.gamma * Q_s_ - q)
        self.set_Q(s, a, q)
        return q

    def get_a(self, s, epsilon):
        if self.policy == 'epsilon_greedy':
            return self.get_a_epsilon_greedy(s, epsilon)
        else:
            raise NotImplementedError

    def get_a_epsilon_greedy(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.env.num_actions)
        else:
            return np.argmax(self.get_Q(s))

    def get_Q(self, s=None, a=None):
        # i, j = self.env.get_pos(s)
        if s is None:
            return self.Q
        else:
            return self.Q[s, :] if a is None else self.Q[s, a]

    def set_Q(self, s, a, value):
        # i, j = self.env.get_pos(s)
        self.Q[s, a] = value

    def get_policy(self, s=None):
        return np.argmax(self.get_Q(s)) if s is not None else np.argmax(self.get_Q(), axis=1)

    def save_Q(self, path):
        np.save(path, self.Q)

    def get_state_value_function(self):
        return np.max(self.get_Q(), axis=1)

    def save_checkpoint(self, path):
        # join path
        path = os.path.join(path, self.scene, self.algorithm)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_values(os.path.join(path, 'Q.npy'))
