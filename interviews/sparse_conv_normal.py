import numpy as np


def sparse_convolution(input_matrix, kernel):
    """
    实现稀疏卷积的函数
    :param input_matrix: 输入的稀疏矩阵
    :param kernel: 卷积核
    :return: 卷积结果
    """
    output_matrix = np.zeros_like(input_matrix)
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_matrix.shape

    # 构建规则手册
    rulebook = []
    for i in range(input_height - kernel_height + 1):
        for j in range(input_width - kernel_width + 1):
            receptive_field = input_matrix[i : i + kernel_height, j : j + kernel_width]
            output_value = np.sum(receptive_field * kernel)
            if output_value != 0:
                rulebook.append((i, j, output_value))

    # 根据规则手册更新输出矩阵
    for i, j, value in rulebook:
        output_matrix[i, j] = value

    return output_matrix


# 示例：创建一个稀疏矩阵和一个卷积核
sparse_matrix = np.array([[0, 1, 0, 0, 0], 
                          [0, 0, 1, 0, 1], 
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0],]
                          )

conv_kernel = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])

# 执行稀疏卷积
result = sparse_convolution(sparse_matrix, conv_kernel)
print("稀疏卷积结果：")
print(result)
