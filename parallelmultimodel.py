import os.path
import tensorflow as tf
import pickle as pkl

from tensorflow import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.layers.merge import concatenate
from matplotlib import pyplot as plt

no_classes = 100
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy


if os.path.isfile("dataset.pkl"):
	model.load("model.h5")

(input_train, target_train), (input_test, target_test) = keras.datasets.cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255


# input layer
visible = Input(shape=input_shape)
# first feature extractor
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv2D(256, kernel_size=8, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)

# third feature extractors 
conv3 = Conv2D(256, kernel_size=8, activation='relu')(visible)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
flat3 = Flatten()(pool3)


# merge feature extractors
merge = concatenate([flat1, flat2, flat3])
# interpretation layer
hidden1 = Dense(128, activation='relu')(merge)
# prediction output
output = Dense(no_classes, activation='softmax')(hidden1)
model = Model(inputs=visible, outputs=output)

# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='img/shared_feature_extractor.png')

#Adam optimizer
opt = keras.optimizers.Adam(lr=0.01)
# Compile the model
model.compile(loss=loss_function,
              optimizer=opt,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, target_train,
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.2)


# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.savefig("img/loss.png")

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.savefig("img/accuracy.png")

model.evaluate(
    x=input_test,
    y=target_test,
    verbose=1
)

model.save("model.h5")

