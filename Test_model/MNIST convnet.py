from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import random

# Best run: 0.9901

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Format images
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32') #use floats so we can scale below
x_test = x_test.astype('float32')
#Scale inputs to 0-1 rather than 0-255
x_train /= 255
x_test /= 255

# Make labels have one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Initialise layers
inputlayer = Input(shape = (28, 28, 1))
hidden = Conv2D(32, 3, strides=(1, 1), activation='tanh')(inputlayer)
hidden = MaxPooling2D()(hidden)
hidden = Conv2D(32, 3, strides=(1, 1), activation='tanh')(hidden)
hidden = MaxPooling2D()(hidden)
outputlayer = Flatten()(hidden)
outputlayer = Dense(100, activation='sigmoid')(outputlayer)
outputlayer = Dense(100, activation='sigmoid')(outputlayer)
outputlayer = Dense(10, activation='sigmoid')(outputlayer)

# Make an instance of the model with the above layers, choose its optimizer/loss etc., and train it
model = Model(inputlayer, outputlayer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)

# Evaluate trained network
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Make lists of which test images were correct and incorrect
predicted_classes = model.predict(x_test) # lists of probabilities of each class
predicted_classes = [np.argmax(example) for example in predicted_classes] # pick out the associated number
y_test = np.argmax(y_test, axis=1) #undo one-hot encoding on y_test, so we can compare
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# See some incorrectly-classes images
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.tight_layout()
random.shuffle(incorrect_indices)
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('Predicted %s, Actual %s' %(predicted_classes[incorrect], y_test[incorrect]))
plt.show()