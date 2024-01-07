"""
普通稀疏卷积：在每一层卷积操作后，输出的稀疏性可能会增加。这意味着即使输入特征图中的某些位置是零，卷积操作仍可能在这些位置生成非零输出。

子流形稀疏卷积：它只在输入特征图中已经是非零的位置上进行卷积操作，保持输出特征图的稀疏性与输入相同。这意味着如果输入特征图在某个位置是零，那么无论卷积核如何，输出在这个位置也将是零。

"""

import numpy as np


def sparse_convolution(input_matrix, kernel, padding=0, stride=1):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 添加padding
    padded_input = np.pad(
        input_matrix, ((padding, padding), (padding, padding)), mode="constant"
    )

    # 计算输出矩阵的大小
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    # 构建规则手册，只针对非零元素
    non_zero_indices = np.argwhere(input_matrix != 0)
    for x, y in non_zero_indices:
        for i in range(kernel_height):
            for j in range(kernel_width):
                # 计算卷积核对应的位置
                row = x + i - padding
                col = y + j - padding
                if 0 <= row < input_height and 0 <= col < input_width:
                    output_row, output_col = row // stride, col // stride
                    output_matrix[output_row, output_col] += (
                        padded_input[row, col] * kernel[i, j]
                    )

    return output_matrix


def submanifold_sparse_convolution(input_matrix, kernel, padding=0, stride=1):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 添加padding
    padded_input = np.pad(
        input_matrix, ((padding, padding), (padding, padding)), mode="constant"
    )

    # 计算输出矩阵的大小
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    # 只在输入矩阵的非零元素位置进行卷积操作
    non_zero_indices = np.argwhere(input_matrix != 0)
    for x, y in non_zero_indices:
        output_row, output_col = (x + padding) // stride, (y + padding) // stride
        for i in range(kernel_height):
            for j in range(kernel_width):
                row = x + i - padding
                col = y + j - padding
                if 0 <= row < input_height and 0 <= col < input_width:
                    output_matrix[output_row, output_col] += (
                        padded_input[row, col] * kernel[i, j]
                    )

    return output_matrix


# 示例
sparse_matrix = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]
)

conv_kernel = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])

padding_size = 1
stride_size = 2
result = sparse_convolution(
    sparse_matrix, conv_kernel, padding=padding_size, stride=stride_size
)
print(f"带有padding和stride的稀疏卷积结果（padding={padding_size}, stride={stride_size}）：")
print(result)


# 示例
result = submanifold_sparse_convolution(
    sparse_matrix, conv_kernel, padding=padding_size, stride=stride_size
)
print(f"子流形稀疏卷积结果（padding={padding_size}, stride={stride_size}）：")
print(result)
