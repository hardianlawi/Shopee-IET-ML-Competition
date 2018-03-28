import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from keras.layers import (
    Activation,
    BatchNormalization,
    Dense, 
    Dropout, 
)
from keras.models import (
    Sequential, 
    load_model,
)
from keras.optimizers import SGD, Adadelta
from keras.utils import to_categorical


def load_data(fname):
    return np.load('../{}.X.npy'.format(fname)), np.load('../{}.y.npy'.format(fname))


def get_lr():
    return LogisticRegression(C=1, penalty='l2', class_weight='balanced')


def get_svm():
    return LinearSVC(C=0.01, penalty='l2', class_weight='balanced')


def get_nn(input_shape, n_classes):
    model = Sequential()

    model.add(Dense(1024, input_shape=input_shape, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(n_classes, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    optimizer = Adadelta()
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    
    return model


# load xception feature vector
X_train, y_train = load_data('Xception-train0')
y_train = np.argmax(y_train, axis=1)
shoes = (y_train == 6) | (y_train == 10)
X_train, y_train = X_train[shoes], y_train[shoes]

X_val, y_val = load_data('Xception-val0')
y_val = np.argmax(y_val, axis=1)
shoes = (y_val == 6) | (y_val == 10)
X_val, y_val = X_val[shoes], y_val[shoes]

input_shape = X_train[0].shape
n_classes = len(set(y_train))
# train model
# model = get_svm()
model = get_lr()
model.fit(X_train, y_train)

# model = get_nn(input_shape, n_classes)
# y_train = np.asarray([1 if x == 6 else 0 for x in y_train])
# y_val = np.asarray([1 if x == 6 else 0 for x in y_val])
# y_train, y_val = to_categorical(y_train), to_categorical(y_val)
# model.fit(X_train, y_train, batch_size=50000, epochs=3, validation_data=(X_val, y_val), verbose=1)

preds = model.predict(X_val)
print('train:', X_train.shape)
print('val:', X_val.shape)
# preds = np.argmax(preds, axis=1)
# y_val = np.argmax(y_val, axis=1)
print('acc =', accuracy_score(preds, y_val))
