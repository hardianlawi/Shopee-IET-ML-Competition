import gc
import os
import json
import numpy as np
import pandas as pd

import keras.backend as K
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

from sklearn.model_selection import StratifiedKFold
from tools.util import load_model, load_images, load_preprocess_input


# Model to use
model_type = "VGG16"
include_top = False
data_augmentation = True
input_shape = (224, 224, 3)

# Filepaths
output_dir = "../outputs"
logs_dir = "../logs"
dataset = "../data/mapTrain.csv"
testDataset = "../data/mapTest.csv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

val_dir = os.path.join(output_dir, "val")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

test_dir = os.path.join(output_dir, "test")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Prepare model saving directory.
model_dir = os.path.join(output_dir, 'saved_models')

if not os.path.isdir(model_dir) or not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Training Config
n_splits = 8  # No of split for skfold cross validation
batch_size = 128  # No of samples fit every step
epochs = 5  # No of epochs

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
df = pd.read_csv(dataset).sample(frac=1.0, random_state=2018)
tDf = pd.read_csv(testDataset)

# number of classes
n_classes = df.category.nunique()

# Load and resize all images
print("Loading images...")
all_imgs, broken_imgs = load_images(df.file, input_shape=input_shape)
test_imgs, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)

# Load base_model
preprocess_input = load_preprocess_input(model_type)

# Convert and process train images to numpy array
X = np.array(all_imgs)  # Matrix of (m x 224 x 224 x 3)
X = preprocess_input(X)
y = df.drop(broken_imgs).as_matrix(columns=["category_id"])  # Convert target to numpy array of m x 1

# Convert and process test images
Xtest = np.array(test_imgs)
Xtest = preprocess_input(Xtest)

# Check shapes
assert X.shape == (len(all_imgs), 224, 224, 3)
assert y.shape == (len(all_imgs), 1)
assert Xtest.shape == (len(test_imgs), 224, 224, 3)

del all_imgs, test_imgs
gc.collect()

# Define optimizer
sgd = optimizers.SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)

# Define a splitter
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)

# Training
print("Start cross-validation training...")
histories = []
valDf = pd.DataFrame()
i = 0
for train, val in skf.split(X, y):

    # Define model name
    model_name = '%s_model_{epoch:03d}_{val_acc:.2f}_iter%d.h5' % (model_type, i)
    filepath = os.path.join(model_dir, model_name)

    # Load model
    model = load_model(model_type, input_shape=input_shape, n_classes=n_classes, include_top=True)

    # compile the model
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    # Prepare callbacks
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True
    )

    lr_reducer = ReduceLROnPlateau(
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=5,
        min_lr=0.5e-6
    )

    csv_logger = CSVLogger(os.path.join(logs_dir, "training.csv"))

    early_stopping = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto'
    )

    callbacks = [checkpoint, lr_reducer, csv_logger, early_stopping]

    # Split training and validation
    Xtrain, ytrain = X[train, :], to_categorical(y[train, :], num_classes=n_classes)
    Xval, yval = X[val, :], to_categorical(y[val, :], num_classes=n_classes)

    if data_augmentation:

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=180,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=None,
            preprocessing_function=None,
            data_format=K.image_data_format()
        )

        history = model.fit_generator(
            datagen.flow(Xtrain, ytrain, batch_size=batch_size),
            validation_data=(Xval, yval),
            epochs=epochs,
            verbose=1,
            workers=16,
            callbacks=callbacks
        )

    else:

        history = model.fit(
            Xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(Xval, yval),
            callbacks=callbacks
        )

    # Generate second-level data
    val_predictions = model.predict(Xval)
    valDf = pd.concat([valDf, pd.DataFrame(val_predictions, index=val, columns=["f"+str(x) for x in range(n_classes)])])

    # Generate prediction on test data for ensembling
    test_predictions = model.predict(Xtest)
    testDf = pd.DataFrame({"id": tDf["id"]})
    testDf = pd.concat([testDf, pd.DataFrame(test_predictions, columns=["f"+str(x) for x in range(n_classes)])], axis=1)
    testDf.to_csv(
        os.path.join(output_dir, "%s_test_iter%d.csv" % (model_type, i)),
        index=False
    )

    i += 1

    histories.append(history)

valDf.to_csv("%s_val.csv" % model_type)

with open(os.path.join(logs_dir, "logs.json"), "w") as f:
    json.dump(histories)
