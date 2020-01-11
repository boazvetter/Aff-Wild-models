"""Data preparation for inference on a convolutional neural network.

Expects png images of faces with N * D dimensions, and converts
this into a tensor of N * N matrices of the right input size (ex. 96x96) for
the neural network. Configured to use with cropped pngs of MTCNN.

Author: Boaz Vetter, 2020
"""
import os
import warnings

import cv2

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import numpy as np


#TODO 0.1 Convert image to square (96x96) image using numpy

def convert_to_square(M, val):
    """Converts the given matrix into a square matrix.

    Pads the given matrix 'M' with value 'val' to change the dimensions
    of the input matrix to square dimensions.

    Args:
      M: 3-D Matrix of type 'numpy.ndarray'.
      val: Scalar value, used to for padding in the square matrix.

    Returns:
      A square matrix, padded with values based on 'value'.
    """
    (row, column, channels) = M.shape
    if row > column:
        one_side = int((row - column)/2)
        padding = ((0, 0), (one_side, one_side), (0,0))
    else:
        one_side = int((column - row)/2)
        padding = ((one_side, one_side), (0, 0), (0,0))
    return np.pad(M, padding, mode='constant', constant_values=val)

dataset_path = os.path.join(os.path.expanduser('~'), 'Documents', 'Thesis', 'Datasets', 'Temp')
img1 = cv2.imread(dataset_path + '/00000.png')
img2 = cv2.imread(dataset_path + '/00001.png')

img1 = convert_to_square(img1, 255)
img1 = cv2.resize(img1, (96,96))

cv2.imwrite('padded_resized.png', img1)


# 1 Convert mtcnn face to tensor

# 2 Decode rgb

# tf_to_float & scale image [-1,1]

# tf.reshape
