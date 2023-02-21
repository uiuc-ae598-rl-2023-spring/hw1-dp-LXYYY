from models.base_model import BaseModel
import numpy as np


class SARSA(BaseModel):
    def __init__(self, env, alpha, epsilon,  gamma=0.95, max_iter=100):
        super().__init__(env, 'sarsa')
        n = int(np.sqrt(self.env.num_states))
        self.Q = np.random.rand(self.env.num_states, self.env.num_actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_a(self, s, epsilon):
        if np.random.random() < epsilon:
            random_a=np.random.randint(0, self.env.num_actions)
            # print(f'random action: {random_a}')
            return np.random.randint(0, self.env.num_actions)
        else:
            return np.argmax(self.get_Q(s))

    def update_Q(self, s, a, r, s_, a_ ,done):
        q = self.get_Q(s, a)
        if done:
            q += self.alpha*(r-q)
        else:
            q += self.alpha*(r+self.gamma*self.get_Q(s_, a_)-q)   
        self.set_Q(s, a, q)
        return q

    def get_Q(self, s, a=None):
        # i, j = self.env.get_pos(s)
        return self.Q[s,:] if a is None else self.Q[s, a]

    def set_Q(self, s, a, value):
        # i, j = self.env.get_pos(s)
        self.Q[s, a] = value

    def get_policy(self, s):
        return np.argmax(self.get_Q(s))

    def get_policy_for_all_s(self):
        return np.argmax(self.Q, axis=1)