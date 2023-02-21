import abc
import numpy as np


# define an abc models class
class BaseModel(abc.ABC):
    class Log:
        def __init__(self):
            self.values = {}
            self.plot_style = {}
            self.plot_args = {}

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

        def plot(self, **kwargs):
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
                    # arrows
                    for i in range(len(v_np) - 1):
                        plt.arrow(v_np[i, 1], v_np[i, 0], v_np[i + 1, 1] - v_np[i, 1], v_np[i + 1, 0] - v_np[i, 0],
                                  head_width=0.2, head_length=0.3, fc='k', ec='k')
                plt.legend()
                plt.show()

    def __init__(self, env):
        self.env = env
        self._values = np.random.rand(int(np.sqrt(self.env.num_states)), int(np.sqrt(self.env.num_states)))
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(0, self.env.num_actions, self._values.shape)
        self.log = self.Log()

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

    def get_mean_value(self):
        return np.mean(self._values)

    def get_log(self):
        return self.log
