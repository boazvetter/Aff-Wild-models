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
        one_side = int((row - column) / 2)
        padding = ((0, 0), (one_side, one_side), (0, 0))
    else:
        one_side = int((column - row) / 2)
        padding = ((one_side, one_side), (0, 0), (0, 0))
    return np.pad(M, padding, mode='constant', constant_values=val)


def get_png_names(folderpath, suffix):
    """Given a path and an exclude string, returns a list of filenames."""
    files = []
    for (path, dirname, filenames) in os.walk(folderpath):
        for file in filenames:
            if '.png' in file and suffix not in file:
                files.append(file)
        break
    files.sort()
    files.sort(key=len)
    return files

def decodeRGB(images, size=96):
    """ Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Three tensors: one with the decoded images, one with the corresponding labels and another with the image file locations
	"""
    images_tensor = []
    for image in images[:5]:
        file_content = tf.read_file(dataset_path+'/'+image)
        image = tf.image.decode_png(file_content, channels=3)
        image = tf.image.resize_images(image, tf.convert_to_tensor([size, size]))
        images_tensor.append(image)

    return images_tensor


dataset_path = os.path.join(os.path.expanduser('~'), 'Documents', 'Thesis', 'Datasets', 'Temp')
suffix = '_cropped'
images = get_png_names(dataset_path, suffix)

for filename in images:
    img = cv2.imread(dataset_path + '/' + filename)
    img = convert_to_square(img, 255)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(dataset_path + '/' + filename[:-4] + suffix + '.png', img)

# TODO 2 Decode jpg rgb
imagetensor = decodeRGB(images, size=96)
imagetensor = tf.to_float(imagetensor)
imagetensor -= 128.0
imagetensor /= 128.0  # scale all pixel values in range: [-1,1]
imagetensor = tf.reshape(imagetensor, [-1, 96, 96, 3])

print(tf.shape(imagetensor))

from vggface import vggface_4096x2000x2 as net
batch_size = 5
network = net.VGGFace(batch_size)


# image_batch is a Tensor of shape (batch_size, 96, 96, 3) containing the images whose pixel intensities are within the range [-1,1]
network.setup(imagetensor)
prediction = network.get_output()
valence_val = prediction[:,0]
arousal_val = prediction[:,1]

print("Valence: ", valence_val)
print("Arousal: ", arousal_val)

with tf.Session() as sess:
    print(valence_val.eval())


# Todo Get inference
#
# variables_to_restore = tf.global_variables()
#
# with tf.Session() as sess:
#     init_fn = slim.assign_from_checkpoint_fn(
#         FLAGS.pretrained_model_checkpoint_path, variables_to_restore,
#         ignore_missing_vars=False)
#
#     init_fn(sess)
#     print('Loading model {}'.format(FLAGS.pretrained_model_checkpoint_path))
#
#     tf.train.start_queue_runners(sess=sess)
#
#     coord = tf.train.Coordinator()
#
#     evaluated_predictions = []
#     evaluated_labels = []
#     images = []
#
#     try:
#         for _ in range(num_batches):
#
#             pr, l, imm = sess.run([prediction, labels_batch, image_locations_batch])
#             evaluated_predictions.append(pr)
#             evaluated_labels.append(l)
#             images.append(imm)
#
#             if coord.should_stop():
#                 break
#         coord.request_stop()
#     except Exception as e:
#         coord.request_stop(e)
#
#     predictions = np.reshape(evaluated_predictions, (-1, 2))
#     labels = np.reshape(evaluated_labels, (-1, 2))
#     images = np.reshape(images, (-1))
#
#     conc_arousal = concordance_cc2(predictions[:, 1], labels[:, 1])
#     conc_valence = concordance_cc2(predictions[:, 0], labels[:, 0])
#
#     print('Concordance on valence : {}'.format(conc_valence))
#     print('Concordance on arousal : {}'.format(conc_arousal))
#     print('Concordance on total : {}'.format((conc_arousal + conc_valence) / 2))
#
#     mse_arousal = sum((predictions[:, 1] - labels[:, 1]) ** 2) / len(labels[:, 1])
#     print('MSE Arousal : {}'.format(mse_arousal))
#     mse_valence = sum((predictions[:, 0] - labels[:, 0]) ** 2) / len(labels[:, 0])
#     print('MSE Valence : {}'.format(mse_valence))
#
# return conc_valence, conc_arousal, (conc_arousal + conc_valence) / 2, mse_arousal, mse_valence
#
#
