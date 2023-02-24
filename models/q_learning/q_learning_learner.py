from models.base_model import ModelFreeAlg
import numpy as np


class QLearning(ModelFreeAlg):
    def __init__(self, env, scene, alpha, epsilon, gamma=0.95):
        super().__init__(env, scene=scene, algorithm='q_learning', alpha=alpha, epsilon=epsilon, gamma=gamma)

    def Q_s_(self, s_, a_):
        assert a_ is None, 'Q-learning does not use a_'
        return np.max(self.get_Q(s_))
