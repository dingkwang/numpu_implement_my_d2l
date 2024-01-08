import torch
import numpy as np

# d2l
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


# gpt4
class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.scale = None
        self.shift = None
        self.moving_mean = None
        self.moving_variance = None
        self.trained = False

    def fit(self, X):
        # 初始化参数
        if self.scale is None:
            self.scale = np.ones(X.shape[1])
            self.shift = np.zeros(X.shape[1])
            self.moving_mean = np.zeros(X.shape[1])
            self.moving_variance = np.ones(X.shape[1])

        # 计算均值和方差
        batch_mean = np.mean(X, axis=0)
        batch_variance = np.var(X, axis=0)

        # 更新移动平均
        self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean
        self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance

        # 归一化
        X_norm = (X - batch_mean) / np.sqrt(batch_variance + self.epsilon)
        # 缩放和位移
        out = self.scale * X_norm + self.shift

        self.trained = True
        return out

    def transform(self, X):
        if not self.trained:
            raise Exception("The BatchNormalization layer must be fitted first.")
        
        # 使用训练时学习的参数进行归一化
        X_norm = (X - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
        out = self.scale * X_norm + self.shift
        return out

# 示例使用
# 创建一个 BatchNormalization 实例
bn = BatchNormalization()

# 假设 X_train 是训练数据
# X_train = ...

# 训练阶段
# normalized_train = bn.fit(X_train)

# 假设 X_test 是测试数据
# X_test = ...

# 测试阶段
# normalized_test = bn.transform(X_test)