import numpy as np
print("IT'S WORKING!!!")
# import tensorflow
import keras
# import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

# train_images = mnist.train_images(-)
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

def isListEmpty(inList):
        if isinstance(inList, list): # Is a list
            return all( map(isListEmpty, inList) )
        return False # Not a list

class NinaproDB:
    winSize = 25
    overlap = 0

    def __init__(self):
        #                               gestures             reps                 subjects            exercises
        self.Data = [ [ [ [ [] for i in range(13) ] for i in range(11) ] for i in range(64)] for i in range(3)]
        return

    def __str__(self):
        gesCount = [0 for i in range(13)]
        subCount = 0

        for exe in self.Data:
            for sub in exe:
                if not isListEmpty(sub):
                    subCount += 1
                    for rep in sub:
                        for ges in range(len(rep)):
                            gesCount[ges] += len(rep[ges])

        string = "%d %s \n" % (subCount, 'subject' if subCount==1 else 'subjects')
        string += "%d windows total\n" % sum(gesCount)
        for ges in gesCount:
            string += "Gesture %d - %d windows\n" % (gesCount.index(ges), ges)
        return string

    def read(self, file):
        dataset = loadmat(file)
        signal = dataset['emg']
        labels = dataset['restimulus']
        exe    = dataset['exercise']
        sub    = dataset['subject']
        rep    = dataset['rerepetition']

        print(labels[0][0])
        print(exe[0][0])
        print(sub[0][0])
        print(rep[0][0])


        # Divide signal into time windows
        # length = (len(signal) // winSize) * winSize

        step = self.winSize - self.overlap
        # self.windows = [signal[i : i + self.winSize] for i in range(0, len(signal), step)]
        # if (self.windows[-1].shape != self.windows[1].shape) :
        #     del self.windows[-1]

        for start in range(0, len(signal) - self.winSize, step):
            end = start + self.winSize
            if labels[start] == labels[end-1]:
                self.Data[exe[0][0]-1][sub[0][0]-1][rep[start][0]-1][labels[start][0]].append(signal[start:end])

            # print(start)
            # print(end)
            # print(exe[0][0])
            # print(sub[0][0])
            # print(rep[start][0])
            # print(labels[start][0])
            # print()




## IMPORT DATA ##
from scipy.io import loadmat
dataset = loadmat('S7_A1_E1.mat')
emg = dataset['emg']
stm = dataset['restimulus']

print(emg.shape)
print(stm.shape)

DB = NinaproDB();
DB.read('S7_A1_E1.mat')


print(DB)

# DB.windows = np.array(DB.windows)
# print(DB.windows.shape)


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


