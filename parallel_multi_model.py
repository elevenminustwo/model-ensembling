# -*- coding: utf-8 -*-

import os.path
import tensorflow as tf
import pickle as pkl
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Dropout
from keras.losses import sparse_categorical_crossentropy
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn.ensemble import VotingClassifier

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
else:
  print('Found GPU at: {}'.format(device_name))

no_classes = 10
img_width, img_height, img_num_channels = 32, 32, 3

(input_train, target_train), (input_test, target_test) = keras.datasets.cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

modelseq = Sequential()
modelseq.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
modelseq.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
modelseq.add(MaxPooling2D((2, 2)))
modelseq.add(Dropout(0.2))
modelseq.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
modelseq.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
modelseq.add(MaxPooling2D((2, 2)))
modelseq.add(Dropout(0.2))
modelseq.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
modelseq.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
modelseq.add(MaxPooling2D((2, 2)))
modelseq.add(Dropout(0.2))
modelseq.add(Flatten())
modelseq.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
modelseq.add(Dropout(0.2))
modelseq.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
modelseq.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# summarize layers
print(modelseq.summary())
# plot graph
#plot_model(modelseq, to_file='modelseq.png')

# first model
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model2.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model2.add(Dropout(0.2))
model2.add(Dense(10, activation='softmax'))

# second model
model3 = Sequential()
model3.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model3.add(BatchNormalization())
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.2))
model3.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.2))
model3.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model3.add(Dropout(0.2))
model3.add(Dense(10, activation='softmax'))

# third model
model4 = Sequential()
model4.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model4.add(BatchNormalization())
model4.add(Dropout(0.2))
model4.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model4.add(BatchNormalization())
model4.add(Dropout(0.2))
model4.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model4.add(BatchNormalization())
model4.add(Dropout(0.2))
model4.add(Flatten())
model4.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model4.add(Dropout(0.2))
model4.add(Dense(10, activation='softmax'))


opt = keras.optimizers.Adam(learning_rate=0.001)
opt2 = keras.optimizers.Adam(learning_rate=0.01)
opt3 = SGD(lr=0.001, momentum=0.9)
model2.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.compile(optimizer=opt2, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.compile(optimizer=opt3, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# summarize layers
print(model2.summary())
print(model3.summary())
print(model4.summary())
# plot graph
#plot_model(model2, to_file='model2.png')
#plot_model(model3, to_file='model3.png')
#plot_model(model4, to_file='model4.png')

"""history = modelseq.fit(input_train, target_train,
            batch_size=64,
            epochs=25,
            verbose=1,
            validation_split=0.2)

history2 = model2.fit(input_train, target_train,
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.2)

history3 = model3.fit(input_train, target_train,
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.2)

history4 = model4.fit(input_train, target_train,
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.2)

# Generate generalization metrics
score = modelseq.evaluate(input_test, target_test, verbose=0)
score2 = model2.evaluate(input_test, target_test, verbose=0)
score3 = model3.evaluate(input_test, target_test, verbose=0)
score4 = model4.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
print(f'Test loss: {score2[0]} / Test accuracy: {score2[1]}')
print(f'Test loss: {score3[0]} / Test accuracy: {score3[1]}')
print(f'Test loss: {score4[0]} / Test accuracy: {score4[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['accuracy'],label = 'train_1')
plt.plot(history.history['val_accuracy'],label = 'test_1')
plt.plot(history2.history['accuracy'],label = 'train_2')
plt.plot(history2.history['val_accuracy'],label = 'test_2')
plt.plot(history3.history['accuracy'],label = 'train_3')
plt.plot(history3.history['val_accuracy'],label = 'test_3')
plt.plot(history4.history['accuracy'],label = 'train_4')
plt.plot(history4.history['val_accuracy'],label = 'test_4')
plt.legend()
plt.show()

#print(input_test[1])
single = np.expand_dims(input_test[250], axis=0)
prediction1 = modelseq.predict(input_test)
prediction2 = model2.predict(input_test)
prediction3 = model3.predict(input_test)
prediction4 = model4.predict(input_test)
"""

scores = list()
scores.append(0.7398999929428101)
scores.append(0.794700026512146)
scores.append(0.6926000118255615)
scores.append(0.6664000153541565)


keras_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = lambda:
                            modelseq,
                            epochs=25,
                            verbose=True)

keras_clf2 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = lambda:
                            model2,
                            epochs=10,
                            verbose=True)
keras_clf3 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = lambda:
                            model3,
                            epochs=10,
                            verbose=True)
keras_clf4 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = lambda:
                            model4,
                            epochs=10,
                            verbose=True)

keras_clf._estimator_type = "classifier"
keras_clf2._estimator_type = "classifier"
keras_clf3._estimator_type = "classifier"
keras_clf4._estimator_type = "classifier"

models = list()	
models.append(('m1', keras_clf))
models.append(('m2', keras_clf2))
models.append(('m3', keras_clf3))
models.append(('m4', keras_clf4))

# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(input_train,target_train.ravel())
# make predictions on test set
yhat = ensemble.predict(input_test)

# evaluate predictions
score = np.mean(yhat==np.squeeze(target_test))
print(scores)
print('Weighted Avg Accuracy: %.3f' % (score*100))
