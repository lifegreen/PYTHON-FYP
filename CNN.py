import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import filtfilt, butter

from datetime import datetime

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class NinaproDB:
    def __init__(self, fs):
        self.winSize = int(0.256 * fs)
        self.step = int(0.064 * fs)
        print('winSize:', self.winSize)
        print('step:', self.step)

        self.fs = fs

        fc = 1  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = butter(5, w, 'low')

        self.Data = {'sub':[], 'exe':[], 'ges':[], 'rep':[], 'win':[]}
        self.subjects  = []
        self.exercises = []
        self.gestures  = []

    def __repr__(self):
        subCount = len(self.subjects)
        exeCount = len(self.exercises)
        gesCount = len(self.gestures)
        winCount = len(self.Data['win'])

        string = self.__class__.__name__ + " with:\n"
        string += "%d %s \n" % (subCount, 'subject' if subCount==1 else 'subjects')
        string += "%d %s \n" % (exeCount, 'excercise' if exeCount==1 else 'excercises')
        string += "%d %s \n" % (gesCount, 'gesture' if gesCount==1 else 'gestures')
        string += "%d %s \n" % (winCount, 'window' if gesCount==1 else 'windows')

        for ges in self.gestures:
            string += "Gesture %d - %d windows\n" % (ges, self.Data['ges'].count(ges))

        return string

    def readDataBase(self, folder, subject, exercise):
        subjects  = r'\d+' if (subject == 'all') else str(subject).replace(', ', '')
        exercises = r'\d' if (exercise == 'all') else str(exercise).replace(', ', '')

        match = r'\AS{}_.*E{}.*\.mat\Z'.format(subjects, exercises)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if re.search(match, file):
                    self.read(os.path.join(root, file))

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])


    def read(self, file):
        print('Reading:', file)
        dataset = loadmat(file)
        signal = dataset['emg']

        # By default this returns list of lists, each with a single scalar.
        # So np.squeeze is used to turn it into a vector.
        ges = np.squeeze(dataset['restimulus'])
        rep = np.squeeze(dataset['rerepetition'])

        # item() is used to return get scalars instead of lists
        exe = dataset['exercise'].item()
        sub = dataset['subject'].item()

        gestStart = next((i for i, x in enumerate(ges) if x!=0)) # returns index of the frist nonzero element

        # Find where the first rest period ends
        restEnd = next((i for i, x in enumerate(ges) if x==2))

        for start in range(gestStart, len(signal) - self.winSize, self.step):
            end = start + self.winSize

            # Make sure the same gesture is performed throught the window duaration
            if (ges[start] != 0) or (end < restEnd):
                if ges[start] == ges[end-1]:
                    self.Data['sub'].append(sub)
                    self.Data['exe'].append(exe)
                    self.Data['rep'].append(rep[start])
                    self.Data['ges'].append(ges[start])
                    self.Data['win'].append(signal[start:end])

    def readMat(self, folder):

        for i in range(9):
            file = 'motion' + str(i) + '.mat'

            print('Reading:', file)
            motion = loadmat(os.path.join(folder, file))['motion' + str(i)]
            motion = motion[~np.all(motion == 0, axis=1)] # remove rows with all zero values

#             for c in range(motion.shape[1]):
#                 motion[:,c] = filtfilt(self.b, self.a, motion[:,c])

            for start in range(0, len(motion) - self.winSize, self.step):
                end = start + self.winSize

                self.Data['ges'].append(i)
                self.Data['win'].append(motion[start:end])

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])

    def splitByRatio(self, ratio):
        [TrainX, ValidX, TrainY, ValidY] = train_test_split(self.Data['win'], self.Data['ges'], test_size=ratio, shuffle=True)
        return [TrainX, TrainY, ValidX, ValidY]

    def splitByCategory(self, name, ls):
        DF = pd.DataFrame(self.Data)

        TrainX = []
        TrainY = []
        ValidX = []
        ValidY = []

        validIdxs = False * len(self.Data)
        for i in ls:
            validIdxs |= (DF[name] == i)

        ValidX = list(DF['win'][validIdxs])
        ValidY = list(DF['ges'][validIdxs])

        TrainX = list(DF['win'][~validIdxs])
        TrainY = list(DF['ges'][~validIdxs])

        return [TrainX, TrainY, ValidX, ValidY]


## IMPORT DATA ##

DB = NinaproDB(2000)
DB.readDataBase(r'C:\Users\Mark\Downloads\Database 2', 'all', 1)

print(DB)

[TrainX, TrainY, ValidX, ValidY] = DB.splitByCategory('rep', [2, 5, 7])
# [TrainX, TrainY, ValidX, ValidY] = DB.splitByRatio(0.33)

lengths = list(map(len, [TrainX, TrainY, ValidX, ValidY]))
print(lengths)
print('Ratio:', lengths[3]/(lengths[3]+lengths[1]))

# Add a colour channel dimension for conv net input
TrainX = [np.expand_dims(x, axis=2) for x in TrainX]
TrainX = np.stack(TrainX, axis=0 )

ValidX = [np.expand_dims(x, axis=2) for x in ValidX]
ValidX = np.stack(ValidX, axis=0 )

print(type(TrainX))
print(TrainX[0].shape)
print(set(ValidY))


## CREATE NETWORK ##
num_filters = 8
filter_size = 3
pool_size = 2
num_classes = len(DB.gestures)
input_shape = TrainX[0].shape
decay = l2(0.0001)

model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=1,
#                  activation='relu',
#                  input_shape=(512,8,1), data_format="channels_last"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))

# model.add(Dense(num_classes, activation='softmax'))

## PAPER 2 ##
decay = l2(0.0005)
model.add(Conv2D(32, kernel_size=(1,input_shape[1]), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=decay))

model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=decay))
model.add(AveragePooling2D(pool_size=3))

model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=decay))
model.add(AveragePooling2D(pool_size=3))

model.add(Conv2D(64, kernel_size=(5,1), activation='relu', padding='same', kernel_regularizer=decay))
model.add(Conv2D(64, kernel_size=1, activation='relu', padding='same', kernel_regularizer=decay))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=decay))

## PAPER 3 ##
# model.add(BatchNormalization(input_shape=input_shape))

# model.add(Conv2D(32, kernel_size=5, activation='relu',  padding='same', input_shape=input_shape, use_bias=False))

# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Dropout(.2))

# model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', use_bias=False))

# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


# model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Flatten())
# model.add(Dropout(.5))
# model.add(Dense(num_classes, activation='softmax'))

model.summary()

#compile model using accuracy to measure model performance
# optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

## TRAIN THE MODEL.##
model.fit(
    TrainX,
    to_categorical(TrainY),
    epochs=30,
    batch_size=64,
    validation_data=(ValidX, to_categorical(ValidY)),
    shuffle=True,
    verbose=2
)


# Predict on the first 5 test images.
predictions = np.argmax(model.predict(ValidX), axis=1)

# Print our model's predictions.
print('Predictions:\t', predictions[:20])

# Check our predictions against the ground truths.
print('Answers:\t\t', ValidY[:20])

correct = predictions[ValidY==predictions]

print('Unique (pred):\t\t', set(predictions))
print('Unique (correct):\t', set(correct))

accuracy = (np.count_nonzero(ValidY==predictions)/ len(ValidY)) * 100
print('Accuracy = %.2f%%' % accuracy)

# Save/Load weights
now = datetime.now() # datetime object containing current date and time
dt_string = now.strftime("Trained Wieghts/({:.0f}%%) HypNetTest %d-%m-%y_%H-%M".format(accuracy))
model.save_weights('{}.h5'.format(dt_string))

cm = confusion_matrix(ValidY, predictions, normalize='true')

print(cm)

plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title('Confusion matrix ')
plt.colorbar()

plt.savefig('{}.png'.format(dt_string))
plt.show()
