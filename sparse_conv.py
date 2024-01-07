import numpy as np
import torch


def sparse_conv2d(input_matrix, kernel, stride=1, padding=0):
    """
    Perform a 2D sparse convolution.

    :param input_matrix: 2D sparse input matrix (numpy array)
    :param kernel: 2D convolution kernel (torch tensor)
    :param stride: Stride of the convolution
    :param padding: Zero-padding added to both sides of the input
    :return: Output matrix after applying sparse convolution
    """
    # Add padding to the input matrix
    input_padded = np.pad(
        input_matrix, pad_width=padding, mode="constant", constant_values=0
    )

    # Create a hash table for non-zero elements
    non_zero_indices = np.argwhere(input_padded != 0)

    # Convert the input matrix to a torch tensor
    input_tensor = torch.from_numpy(input_padded).float()

    # Initialize the output matrix
    output_shape = (
        (input_tensor.shape[0] - kernel.shape[0]) // stride + 1,
        (input_tensor.shape[1] - kernel.shape[1]) // stride + 1,
    )
    output_matrix = torch.zeros(output_shape)

    # Perform the convolution at non-zero positions
    for i, j in non_zero_indices:
        for y in range(kernel.shape[0]):
            for x in range(kernel.shape[1]):
                if (i + y < input_tensor.shape[0]) and (j + x < input_tensor.shape[1]):
                    output_matrix[i // stride, j // stride] += (
                        input_tensor[i + y, j + x] * kernel[y, x]
                    )

    return output_matrix


# Example usage
input_matrix = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
kernel = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
output = sparse_conv2d(input_matrix, kernel, stride=1, padding=1)
print(output)
