import gc
import os
import json
import numpy as np
import pandas as pd

import keras.backend as K

from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler

from sklearn.model_selection import StratifiedKFold
from tools.util import load_model, load_images, load_preprocess_input, scheduler, LearningRateTracker


np.random.seed(2018)
K.clear_session()

# Model to use
model_type = "ResNet50"
include_top = True
stack_new_layers = True
data_augmentation = True
input_shape = (224, 224, 3)
dropout_rate = 0.5

# Session Name
SESSION_NAME = 'trykevin1'

# Filepaths
output_dir = "../outputs/{}".format(SESSION_NAME)
logs_dir = "../logs/{}".format(SESSION_NAME)
dataset = "../data/{}/mapTrain.csv".format(SESSION_NAME)
testDataset = "../data/{}/mapTest.csv".format(SESSION_NAME)

# dataset is a csv file of format below:
# file,category,category_id
# ../train/Baby/Baby_001.jpg,Baby,1

# testDataset is a csv file of format below:
# id,file
# 1, ../test/Test_001.jpg

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
n_splits = 7  # No of split for skfold cross validation
batch_size = 64  # No of samples fit every step
epochs = 50  # No of epochs
lr = 0.01  # Optimizer learning rate

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
print(dataset)
df = pd.read_csv(dataset).sample(frac=1, random_state=2018)
tDf = pd.read_csv(testDataset)

# number of classes
n_classes = df.category.nunique()

# Load and resize all images
print("Loading images...")
all_imgs, broken_imgs = load_images(df.file, input_shape=input_shape)
test_imgs, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)

# Load preprocessor based on model_type
preprocess_input = load_preprocess_input(model_type)

# Convert and process train images to numpy array
X = np.array(all_imgs)  # Matrix of (m x input_shape)
y = df.drop(broken_imgs).as_matrix(columns=["category_id"])  # Convert target to numpy array of m x 1

# Convert and process test images
Xtest = np.array(test_imgs)
Xtest = preprocess_input(Xtest)

# Check shapes
assert X.shape == (len(all_imgs),) + input_shape
assert y.shape == (len(all_imgs), 1)
assert Xtest.shape == (len(test_imgs),) + input_shape

del all_imgs, test_imgs
gc.collect()

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
    model = load_model(
        model_type,
        input_shape=input_shape,
        n_classes=n_classes,
        include_top=include_top,
        stack_new_layers=stack_new_layers,
        dropout_rate=dropout_rate
    )

    # Define optimizer
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9)

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
        monitor="val_acc",
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=2,
        min_lr=0.5e-6
    )

    csv_logger = CSVLogger(os.path.join(logs_dir, "%s_training_iter%d.csv" % (model_type, i)))

    early_stopping = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto'
    )

    lr_scheduler = LearningRateScheduler(scheduler)

    lr_tracker = LearningRateTracker()

    callbacks = [checkpoint, lr_reducer, csv_logger, early_stopping, lr_scheduler, lr_tracker]

    # Split training and validation
    Xtrain, ytrain = preprocess_input(X[train, :]), to_categorical(y[train, :], num_classes=n_classes)
    Xval, yval = preprocess_input(X[val, :]), to_categorical(y[val, :], num_classes=n_classes)

    if data_augmentation:

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
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
            callbacks=callbacks,
        )

    else:

        history = model.fit(
            Xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(Xval, yval),
            callbacks=callbacks
        )

    del Xtrain, ytrain
    gc.collect()

    # Generate second-level data
    val_predictions = model.predict(Xval, verbose=1)
    valDf = pd.concat([
        valDf,
        pd.DataFrame(np.hstack([val_predictions, y[val, :]]), index=val, columns=["f"+str(x) for x in range(n_classes)] + ["category_id"])])

    del Xval, yval
    gc.collect()

    # Generate prediction on test data for ensembling
    test_predictions = model.predict(Xtest, verbose=1)
    testDf = pd.DataFrame({"id": tDf["id"]})
    testDf = pd.concat([testDf, pd.DataFrame(test_predictions, columns=["f"+str(x) for x in range(n_classes)])], axis=1)
    testDf.to_csv(
        os.path.join(test_dir, "%s_test_iter%d.csv" % (model_type, i)),
        index=False
    )

    i += 1

    histories.append(history)

    del val_predictions, test_predictions, testDf
    gc.collect()

    K.clear_session()

valDf.to_csv(os.path.join(val_dir, "%s_val.csv" % model_type))

with open(os.path.join(logs_dir, "logs.json"), "w") as f:
    json.dump(histories)
