from keras.models import Sequential
import h5py
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD 
def make_network():
    model = Sequential()
    model.add(Convolution2D(64, 11, 11,border_mode='valid',input_shape=(3,224,224)))
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
    model.add(Dense(15))
    model.add(Activation('softmax'))
    return model


def save_model(model):
 
   model_json = model.to_json()
   open('alex_architecture.json', 'w').write(model_json)
   model.save_weights('alex_weights.h5', overwrite=True)
   

if __name__=="__main__":
    model = make_network()
    sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True )
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    #save_model(model)
    #for layer in model.layers:
    #    print(layer.get_weights())
    print(model.summary())
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')
    print(model.get_config())
    print("SAVED ########")
    #import numpy as np
    #imgX = np.random.rand(100,3,224,224)
    #imgY = np.random.rand(100,15)
    import scipy.io as sio
    all_test = sio.loadmat('testData.mat')
    from sklearn.metrics import confusion_matrix
    X_test = all_test['X_test']
    Y_test = all_test['Y_test']
    X_test = X_test.reshape(X_test.shape[0], 3,224,224)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_pred = model.predict(X_test)
    #Y_pred = model.predict(X_test,batch_size=2,verbose=1)
    print(Y_pred)
    import keras_cnn 
    #Y_one = keras_cnn.oneShot(all_test['Y_test'])
    #print(Y_one)
    #print(confusion_matrix(Y_one,Y_pred))
    #print("Saved Model")




