import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import random
import tensorflow as tf
import keras

#load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#visualize some images
i = 3000
plt.imshow(X_train[i])
print(y_train[i])
plt.show()

W_grid = 5
L_grid = 5

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.4)
plt.show()

#type conversion because we need to normalize the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_cat = 10
y_test = keras.utils.to_categorical(y_test, number_cat)
y_train = keras.utils.to_categorical(y_train, number_cat)

#normalize the data
X_train = X_train / 255
X_test = X_test / 255

print(X_train.shape, X_test.shape)

input_shape = X_train.shape[1:]

#build the model
from keras.models import Sequential

cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 1024, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Dense(units = 1024, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

print(cnn.summary())

#compile the model
optimizer_1 = tf.keras.optimizers.Adam(learning_rate = 0.001)
optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate = 0.001, decay = 1e-6)
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizer_1, metrics = ['accuracy'])

#train the model
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data = (X_test, y_test))


#evaluate the model
evaluation = cnn.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

#predict the classes
predicted_classes = np.argmax(cnn.predict(X_test), axis=-1)
print(predicted_classes)

#convert the classes to the original form
y_test = y_test.argmax(1)

#plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
print(cm)

L = 10
W = 10
fig, axes = plt.subplots(L, W, figsize = (25,25))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)
plt.show()




