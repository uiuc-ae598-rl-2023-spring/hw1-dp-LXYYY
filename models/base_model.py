import abc
import numpy as np


# define an abc models class
class BaseModel(abc.ABC):
    class Log:
        def __init__(self, scene, experiment):
            self.values = {}
            self.plot_style = {}
            self.plot_args = {}
            self.experiment = experiment
            self.scene = scene

        def add(self, key, value, plot=None, **kwargs):
            if key not in self.values:
                self.values[key] = []
                self.plot_style[key] = None
                self.plot_args[key] = {}
            self.values[key].append(value)
            if plot is not None:
                self.plot_style[key] = plot
            if kwargs:
                self.plot_args[key] = kwargs

        def plot(self, save=False, **kwargs):
            import matplotlib.pyplot as plt
            for k, v in self.values.items():
                plt.figure()
                if self.plot_style[k] is None:
                    plt.plot(v, label=k, **kwargs, **self.plot_args[k])
                elif self.plot_style[k] == 'trajectory':
                    # set plot x limit from -1 to 5
                    plt.xlim(-1, 5)
                    # set plot y limit from -1 to 5
                    plt.ylim(-1, 5)
                    # generate x and y coordinates grid
                    x = np.linspace(0, 4, 5)
                    X, Y = np.meshgrid(x, x)
                    # draw grid
                    plt.scatter(X, Y, marker='s', c='k')

                    v_np = np.array(v)
                    v_np[:, 0] = 4 - v_np[:, 0]
                    # arrows
                    for i in range(len(v_np) - 1):
                        plt.arrow(v_np[i, 1], v_np[i, 0],
                                  v_np[i + 1, 1] -
                                  v_np[i, 1], v_np[i + 1, 0] - v_np[i, 0],
                                  head_width=0.2, head_length=0.3, fc='k', ec='k')
                plt.legend()
                if save:
                    plt.savefig('figures/' + self.scene + '/' + self.experiment + '_' + k + '.png')
                plt.show()

    def __init__(self, env, scene, experiment):
        self.env = env
        self._values = np.random.rand(self.env.num_states)
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(
            0, self.env.num_actions, self._values.shape)
        self.log = self.Log(scene=scene, experiment=experiment)

    def get_pos(self, s):
        return s

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

    def get_mean_value(self):
        return np.mean(self._values)

    def get_log(self):
        return self.log


class ModelFreeAlg(BaseModel):
    def __init__(self, env, scene, experiment, alpha, epsilon, gamma=0.95):
        super().__init__(env, scene, experiment)
        self.Q = np.random.rand(self.env.num_states, self.env.num_actions)
        # self.Q = np.zeros([self.env.num_states, self.env.num_actions], dtype=np.float32)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.policy = 'epsilon_greedy'
        self.value_learning_method = 'TD0'

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

    def get_Q(self, s, a=None):
        # i, j = self.env.get_pos(s)
        return self.Q[s, :] if a is None else self.Q[s, a]

    def set_Q(self, s, a, value):
        # i, j = self.env.get_pos(s)
        self.Q[s, a] = value

    def get_policy(self, s):
        return np.argmax(self.get_Q(s))

    def get_policy_for_all_s(self):
        return np.argmax(self.Q, axis=1)
