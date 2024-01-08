import torch
import torch.nn.functional as F


def conv2d(input, in_channels, out_channels, kernel_size, stride=1, padding=0):
    # Initialize weights and bias
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels)

    # Apply padding to the input
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    # Calculate output dimensions
    _, _, height, width = input.shape
    output_height = (height - kernel_size + 2 * padding) // stride + 1
    output_width = (width - kernel_size + 2 * padding) // stride + 1

    # Initialize output tensor
    output = torch.zeros((input.shape[0], out_channels, output_height, output_width))

    # Perform convolution
    for y in range(output_height):
        for x in range(output_width):
            # Extract the region of interest
            y_start = y * stride
            y_end = y_start + kernel_size
            x_start = x * stride
            x_end = x_start + kernel_size

            region = input[:, :, y_start:y_end, x_start:x_end]
            output[:, :, y, x] = torch.sum(region * weight, dim=(1, 2, 3))

    # Add bias
    output += bias.view(1, -1, 1, 1)

    return output

# Example usage
batch_size, in_channels, height, width = 1, 1, 5, 5
out_channels, kernel_size, stride, padding = 1, 3, 1, 1
input = torch.rand(batch_size, in_channels, height, width)

output = conv2d(input, in_channels, out_channels, kernel_size, stride, padding)
output.shape
