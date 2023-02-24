import abc
import numpy as np
import os


# define an abc models class
class BaseModel(abc.ABC):
    class Plot:
        def __init__(self, env, scene, experiment):
            self.values = {}
            self.plot_style = {}
            self.plot_args = {}
            self.experiment = experiment
            self.scene = scene
            self.env = env

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

        def _plot_grid(self):
            import matplotlib.pyplot as plt
            # set plot x limit from -1 to state shape 0
            plt.xlim(-1, self.env.state_shape[0])
            # set plot y limit from -1 to state shape 1
            plt.ylim(-1, self.env.state_shape[1])
            # generate x and y coordinates grid
            x = np.linspace(0, self.env.state_shape[0] - 1, self.env.state_shape[0])
            X, Y = np.meshgrid(x, x)
            # draw grid
            plt.scatter(X, Y, marker='s', c='k')

        def plot(self, save=False, **kwargs):
            import matplotlib.pyplot as plt
            for k, v in self.values.items():
                plt.figure()
                if self.plot_style[k] is None:
                    plt.plot(v, label=k, **kwargs, **self.plot_args[k])
                elif self.plot_style[k] == 'trajectory':
                    self._plot_grid()
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

        def plot_policy(self, policy, save=False):
            if self.scene == 'gridworld':
                import matplotlib.pyplot as plt
                self._plot_grid()
                # arrows
                for i in range(len(policy) - 1):
                    x, y = self.env.get_pos(i)
                    # draw policy as arrows, 0: right, 1: up, 2: left, 3: down
                    l = 0.3
                    dxdy = [[l, 0], [0, l], [-l, 0], [0, -l]]
                    plt.arrow(y, 4 - x, dxdy[policy[i]][0], dxdy[policy[i]][1], head_width=0.1, head_length=0.1, fc='k',
                              ec='k')
                if save:
                    plt.savefig('figures/' + self.scene + '/' + self.experiment + '_policy.png')
                plt.show()
            elif self.scene == 'pendulum':
                import matplotlib.pyplot as plt
                self._plot_grid()
                # arrows
                for i in range(len(policy) - 1):
                    plt.arrow(policy[i, 1], policy[i, 0],
                              policy[i + 1, 1] -
                              policy[i, 1], policy[i + 1, 0] - policy[i, 0],
                              head_width=0.2, head_length=0.3, fc='k', ec='k')
                if save:
                    plt.savefig('figures/' + self.scene + '/' + 'policy.png')
                plt.show()

        def plot_state_value_function(self, V, save=False):
            import matplotlib.pyplot as plt
            # colormap from q values to colors
            cmap = plt.get_cmap('jet')
            # get max and min q values
            max_v_min = np.min(V)
            max_v_max = np.max(V)
            self._plot_grid()
            # draw q values as colors
            for i in range(len(V)):
                x, y = self.env.get_pos(i)
                # draw boxes and set size to a grid box without gap, and shape to square
                plt.scatter(y, 4 - x, marker='s', c=cmap((V[i] - max_v_min) / (max_v_max - max_v_min)), s=1000)

            # draw colorbar and set max and min values
            cbar = plt.colorbar()
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels([max_v_min, max_v_max])
            if save:
                plt.savefig('figures/' + self.scene + '/' + self.experiment + '_value_function.png')
            plt.show()

    def __init__(self, env, scene, algorithm):
        self.env = env
        self._values = np.random.rand(self.env.num_states)
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(
            0, self.env.num_actions, self._values.shape)
        self.plot = self.Plot(env=env, scene=scene, experiment=algorithm)
        self.algorithm = algorithm
        self.scene = scene

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
    class Plot(BaseModel.Plot):
        def __init__(self, env, scene, experiment):
            super().__init__(env, scene, experiment)

    def __init__(self, env, scene, algorithm, alpha, epsilon, gamma=0.95):
        super().__init__(env, scene, algorithm)
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
