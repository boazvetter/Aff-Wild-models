# import vggface_4096x2000x2 as net
# import tensorflow as tf
import numpy as np
import cv2
import os

# TODO: Create the right kind of input tensor
datasetpath = os.path.join(os.path.expanduser('~'), 'Documents', 'Thesis', 'Datasets', 'Temp')
img1 = cv2.imread(datasetpath+'/00000.png')
img2 = cv2.imread(datasetpath+'/00001.png')




# network = net.VGGFace(batch_size)
#
# ## image_batch is a Tensor of shape (batch_size, 96, 96, 3) containing the images whose pixel intensities are within the range [-1,1]
#
# network.setup(image_batch)
# prediction = network.get_output()
# valence_val = prediction[:,0]
# arousal_val = prediction[:,1]
