# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 05:38:15 2019
Script to test
@author: Tudor Trita
"""

from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import imageio

#Parameters:
pixels1 = 100
pixels2 = 100

#No of test images
test_im_no = 330
test_type1 = 164     # Test images of type 1
test_type1_loc = "testdata/Apple Red 1/"
test_type2 = 166     # Test images of type 2
test_type2_loc = "testdata/Banana/"

#No of colours (B/W: 1, RGB: 3)
colours = 3

#Loading test images:
imtest = np.zeros((test_im_no,pixels1,pixels2,colours))

for i in range(test_type1):
    imtest[i] = imageio.imread(test_type1_loc + str(i+1) + ".jpg")
for i in range(test_type2):
    imtest[i+test_type1] = imageio.imread(test_type2_loc + str(i+1) + ".jpg")

#Defining y_test
y_test = np.zeros(test_im_no)
y_test[test_type2:] = 1

# Load data
(x_test, y_test) = (imtest, y_test)

#Format images
x_test = x_test.reshape(test_im_no, pixels1, pixels2, colours)
x_test = x_test.astype('float32')

#Scale inputs to 0-1 rather than 0-255
x_test /= 255

# Make labels have one-hot encoding
y_test = np_utils.to_categorical(y_test, 2)


#Load model:

import_dir = 'C:/Users/Tudor Trita/Desktop/ichack2019/AI_Backend/ModelSaved1'

#Starting a tensorflow session with global variables
sess = tf.InteractiveSession()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_g)
sess.run(init_l)

with tf.Session(graph=tf.Graph()) as sess:
    model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], import_dir)

    # Evaluate trained network
    score = model.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score[0])
print('Test accuracy:', score[1])

sess.close()























