import numpy as np
print("IT'S WORKING!!!")
# import tensorflow
import keras
# import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

# train_images = mnist.train_images()
# train_labels = mnist.train_labels()
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

# # Normalize the images.
# train_images = (train_images / 255) - 0.5
# test_images = (test_images / 255) - 0.5

# # Reshape the images.
# train_images = np.expand_dims(train_images, axis=3)
# test_images = np.expand_dims(test_images, axis=3)

# print(train_images.shape) # (60000, 28, 28, 1)
# print(test_images.shape)  # (10000, 28, 28, 1)

class NinaproDB:
    winSize = 25
    overlap = 0

    def __init__(self):

        return

    def read(self, file):
        dataset = loadmat(file)
        signal   = dataset['emg']
        labels   = dataset['restimulus']
        subject  = dataset['subject']
        exercise = dataset['exercise']

        # Divide signal into time windows
        # length = (len(signal) // winSize) * winSize

        step = self.winSize - self.overlap
        # self.windows = [signal[i : i + self.winSize] for i in range(0, len(signal), step)]
        # if (self.windows[-1].shape != self.windows[1].shape) :
        #     del self.windows[-1]

        # self.windows = []
        # self.classes = []
        for start in range(0, len(signal) - self.winSize, step):
            end = start + self.winSize

            if labels[start] == labels[end-1]:
                self.windows.append(signal[start:end])
                self.classes.append(labels[start])

        # classes = []
        # for i in range(len(windows)):
        #   if
        #   classes.appened()



## IMPORT DATA ##
from scipy.io import loadmat
dataset = loadmat('S7_A1_E1.mat')
emg = dataset['emg']
stm = dataset['restimulus']

print(emg.shape)
print(stm.shape)

print(type(emg))
print(type(stm))

DB = NinaproDB();
DB.read('S7_A1_E1.mat')

print(len(DB.windows))
print(DB.windows[-1].shape)

DB.windows = np.array(DB.windows)
print(DB.windows.shape)


## CREATE NETWORK ##
num_filters = 8
filter_size = 3
pool_size = 2
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

print(model)

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## TRAIN THE MODEL.##
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
)


# Save/Load weights
# model.save_weights('cnn.h5')
# model.load_weights('cnn.h5')


# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]


