from models.base_model import ModelFreeAlg
import numpy as np


class SARSA(ModelFreeAlg):
    def __init__(self, env, alpha, epsilon,  gamma=0.95):
        super().__init__(env, 'sarsa', alpha=alpha, epsilon=epsilon, gamma=gamma)

    def Q_s_(self, s_, a_):
        return self.get_Q(s_, a_)

