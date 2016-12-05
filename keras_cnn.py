
from __future__ import print_function
import numpy as np
import data
from keras.layers.normalization import BatchNormalization
np.random.seed(1337)  # for reproducibility
#import model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import psData as data
import pdb
import scipy.io as sio
import h5py
#Set up intial parameters
batch_size = 8
nb_classes = 15
nb_epoch = 1
img_rows= 224
img_cols = 224



def save_model(model,epoch):
 
   model_json = model.to_json()
   open('Alexnet_model_' + str(epoch) +'.json', 'w').write(model_json)
   model.save_weights('Alexnet_model_'+str(epoch)+'.h5', overwrite=True)


def oneShot(yOut):
    """
    """
    oneClass = np.empty(yOut.shape[0])
    samples = yOut.shape[0]
    for i in range(samples):
        oneClass[i]=np.argmax(yOut[i])

    return oneClass

#Import data 
#X_train,Y_train = data.getTrainData()

#Shuffle the data
#ranIdxTr = np.random.randint(X_train.shape[0],size=X_train.shape[0])
#X_train = X_train[ranIdxTr]
#Y_train = Y_train[ranIdxTr]

#X_test,Y_test = data.getTestData()
#allTrain = sio.loadmat('testData.mat')
allTest = sio.loadmat('testData.mat')

#X_train = allTrain['X_test']
#Y_train = allTrain['Y_test']

X_test = allTest['X_test']
Y_test = allTest['Y_test']


X_train = allTest['X_test']
Y_train = allTest['Y_test']


ranIdxTe = np.random.randint(X_test.shape[0],size = X_test.shape[0])
X_test = X_test[ranIdxTe]
Y_test = Y_test[ranIdxTe]
 
# Reshape data as the model accepts 
X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Define model. 
model = Sequential()
model.add(Convolution2D(64, 11, 11,border_mode='valid',input_shape=(3, img_rows, img_cols)))
model.add(BatchNormalization(64))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(128,7,7,border_mode='valid'))
model.add(BatchNormalization(128))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(192,3,3,border_mode='valid'))
model.add(BatchNormalization(192))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(256,3,3,border_mode='valid'))
model.add(BatchNormalization(256))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4096))
model.add(BatchNormalization(4096))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(4096))
model.add(BatchNormalization(4096))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#print(model)

sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True )
#Set up optimizer and loss function 
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


#pdb.set_trace()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))


#save_model(model,12)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(model.summary())

Y_pred = model.predict_classes(X_test, batch_size=8, verbose=1)
Y_testShot = oneShot(Y_test)
print("YSHOT", Y_testShot)
sio.savemat('Y_true.mat',{'Y_testShot':Y_testShot})
print ("YPRED",Y_pred)
sio.savemat('Y_pred.mat',{'Y_pred':Y_pred})
#Print confusion matrix
from sklearn.metrics import confusion_matrix
con_matrix = confusion_matrix(Y_testShot,Y_pred)
print(con_matrix)

