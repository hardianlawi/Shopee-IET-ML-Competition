"""
Generator with 10 splits
"""


import gc
import os
import numpy as np
import pandas as pd

import keras.backend as K

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler

from tools.util import load_model, load_images, load_preprocess_input, scheduler, LearningRateTracker


np.random.seed(2018)
K.clear_session()

# Model to use
model_type = "DenseNet169"
include_top = True
stack_new_layers = True
input_shape = (224, 224)
dropout_rate = 0.5
n_classes = 18

# Filepaths
output_dir = "../outputs"
logs_dir = "../logs"
train_val_dir = "../data/train_val_v2"
testDataset = "../meta/mapTest.csv"

# dataset is a csv file of format below:
# file,category,category_id
# ../train/Baby/Baby_001.jpg,Baby,1

# testDataset is a csv file of format below:
# id,file
# 1, ../test/Test_001.jpg

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

val_dir = os.path.join(output_dir, "val_v2")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

test_dir = os.path.join(output_dir, "test_v2")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Prepare model saving directory.
model_dir = os.path.join(output_dir, 'saved_models_v2')

if not os.path.isdir(model_dir) or not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Training Config
n_splits = 10  # No of split for skfold cross validation
batch_size = 32  # No of samples fit every step
epochs = 50  # No of epochs
lr = 0.005  # Optimizer learning rate

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
tDf = pd.read_csv(testDataset)

# Load test images
test_imgs, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)

# Load preprocessor based on model_type
preprocess_input = load_preprocess_input(model_type)

# Convert and process test images
Xtest = np.array(test_imgs)
Xtest = preprocess_input(Xtest)

# Check shapes
assert Xtest.shape == (len(test_imgs),) + input_shape + (3, )

del test_imgs
gc.collect()

# Training
print("Start cross-validation training...")
histories = []
valDf = pd.DataFrame()
for iteration in range(n_splits):

    # Define model name
    model_name = '%s_model_{epoch:03d}_{val_acc:.2f}_iter%d.h5' % (model_type, iteration)
    filepath = os.path.join(model_dir, model_name)

    # Load model
    model = load_model(
        model_type,
        input_shape=input_shape + (3,),
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
        patience=1,
        min_lr=0.5e-6
    )

    csv_logger = CSVLogger(os.path.join(logs_dir, "%s_training_iter%d.csv" % (model_type, iteration)))

    early_stopping = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=4,
        verbose=0,
        mode='auto'
    )

    lr_scheduler = LearningRateScheduler(scheduler)

    lr_tracker = LearningRateTracker()

    callbacks = [checkpoint, lr_reducer, csv_logger, early_stopping, lr_scheduler, lr_tracker]

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
        preprocessing_function=preprocess_input,
        data_format=K.image_data_format()
    )

    train_generator = datagen.flow_from_directory(
        directory=os.path.join(train_val_dir, "train_val_%d" % iteration, "train"),
        target_size=input_shape,
        class_mode="categorical",
        batch_size=batch_size,
        seed=2018
    )

    train_val_generator = datagen.flow_from_directory(
        directory=os.path.join(train_val_dir, "train_val_%d" % iteration, "val"),
        target_size=input_shape,
        class_mode="categorical",
        batch_size=batch_size,
        seed=2018
    )

    history = model.fit_generator(
        generator=train_generator,
        validation_data=train_val_generator,
        epochs=epochs,
        verbose=1,
        workers=8,
        callbacks=callbacks,
    )

    val_generator = datagen.flow_from_directory(
        directory=os.path.join(train_val_dir, "train_val_%d" % iteration, "val"),
        target_size=input_shape,
        class_mode="categorical",
        batch_size=100000,
        seed=2018
    )

    # Generate second-level data
    print("Generating second-level data...")
    for Xval, label in val_generator:
        val_predictions = model.predict(Xval, verbose=1)
        valDf = pd.concat([
            valDf,
            pd.DataFrame(
                np.hstack([
                    val_predictions,
                    np.argmax(label, axis=-1)[:, np.newaxis]
                ]),
                columns=["f"+str(x) for x in range(n_classes)] + ["category_id"])
        ])
        break

    del Xval, label, val_predictions
    gc.collect()

    # Generate prediction on test data for ensembling
    print("Generating prediction on test data...")
    test_predictions = model.predict(Xtest, verbose=1)
    testDf = pd.DataFrame({"id": tDf["id"]})
    testDf = pd.concat([testDf, pd.DataFrame(test_predictions, columns=[model_type+"_f"+str(x) for x in range(n_classes)])], axis=1)
    testDf.to_csv(
        os.path.join(test_dir, "%s_test_iter%d.csv" % (model_type, iteration)),
        index=False
    )

    histories.append(history)

    del test_predictions, testDf
    gc.collect()

    K.clear_session()

valDf.to_csv(os.path.join(val_dir, "%s_val.csv" % model_type), index=False)