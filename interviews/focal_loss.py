import numpy as np


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    计算 Focal Loss

    参数:
    y_true: 真实标签，形状为 (n_samples,)
    y_pred: 预测概率，形状为 (n_samples,)
    alpha: 正样本的权重
    gamma: 调节参数

    返回:
    loss: 计算得到的 Focal Loss
    """
    # 保证 y_pred 在一个安全的范围内，以避免数值不稳定
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

    # 计算正类和负类的 Focal Loss
    loss_pos = -alpha * (1 - y_pred) ** gamma * y_true * np.log(y_pred)
    loss_neg = -(1 - alpha) * y_pred ** gamma * (1 - y_true) * np.log(1 - y_pred)

    # 总的 Focal Loss
    loss = np.mean(loss_pos + loss_neg)
    return loss

# 示例
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.2])

loss = focal_loss(y_true, y_pred)
print("Focal Loss:", loss)