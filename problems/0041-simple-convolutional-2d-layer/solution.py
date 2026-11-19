import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # pad the input
    padded_input = np.zeros((input_height + 2*padding, input_width + 2*padding))
    padded_input[padding: input_height+padding, padding: input_width+padding] = input_matrix

    # print(padded_input)
    # do kernel operation
    out_h = (input_height + 2*padding - kernel_height)//stride + 1
    out_w = (input_width + 2*padding - kernel_width)//stride + 1
    output_matrix = np.zeros((out_h, out_w))
    # print((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            region = padded_input[r*stride:r*stride+kernel_height, c*stride:c*stride+kernel_width]
            output_matrix[r, c] = np.sum(region*kernel)
    
    return output_matrix
