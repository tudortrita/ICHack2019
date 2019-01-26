# Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
import os

# Data with one-hot encoding
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Review some images. Set up the plot:
fig = plt.figure(1, (5., 5.))
plt.clf()
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.3)

# Choose, reshape, and plot 9 images
indices = [randint(0, len(mnist.train.images)) for k in range(9)]
for i in range(9):
    image = mnist.train.images[indices[i]].reshape(28,28) # Reshape list into square array
    grid[i].imshow(image) # Add to plot
    grid[i].set_title('Label: {0}'.format(mnist.train.labels[indices[i]].argmax())) # Title and label
fig.suptitle('MNIST image samples')
plt.show()

# To neatly handle global variables later
sess = tf.InteractiveSession()

# Shorthand for Tensorflow weights, biases, convolutions, and pooling
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Create placeholders for images and label inputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Reshape input layer so we can apply convolutions
x_image = tf.reshape(x, [-1,28,28,1])

# Conv layer 1 - 32 5x5 filters
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
x_pool1 = max_pooling_2x2(x_conv1)

# Conv layer 2 - 64 5x5 filters
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
x_conv2 = tf.nn.relu(conv2d(x_pool1, W_conv2) + b_conv2)
x_pool2 = max_pooling_2x2(x_conv2)

# Flatten ready for dense layer
x_flat = tf.reshape(x_pool2, [-1, 7*7*64])

# Dense fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # max pooling reduced image to 7x7
b_fc1 = bias_variable([1024])
x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# Regularization with dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer and output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2
y = tf.nn.softmax(y_conv)

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Setup to test accuracy later
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all global variables (Tensorflow handles this for us)
sess.run(tf.global_variables_initializer())

# Train model
for i in range(5):
    batch = mnist.train.next_batch(100)
    if i%5 == 0: # Show accuracy every 5 batches
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})






#SAVE MODEL
export_path = 'C:/Users/Christophe/Documents/Imperial/UROP/SavedModelFolder.py'
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING]
#      ,signature_def_map={'predict_images':prediction_signature, signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,},main_op=main_op
      )
builder.save()













# Run trained model against test data
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))

sess.close()