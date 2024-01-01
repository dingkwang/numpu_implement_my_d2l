import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update_params(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        params_updated = []

        for p, dp, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * dp
            v[:] = self.beta2 * v + (1 - self.beta2) * (dp ** 2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p_updated = p - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            params_updated.append(p_updated)

        return params_updated
