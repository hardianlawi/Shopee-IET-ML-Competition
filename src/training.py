import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler

from tools.util import load_model, load_images, load_preprocess_input, scheduler, LearningRateTracker, ModelMGPU, load_default_input_shape


np.random.seed(2018)
K.clear_session()

# Filepaths
output_dir = "../outputs"
logs_dir = "../logs"
train_val_dir = "../data/train_val_v1"
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


def main(model_type):

    # Load default input shape and preprocess_input
    input_shape = load_default_input_shape(model_type)
    preprocess_input = load_preprocess_input(model_type)

    # Configs
    include_top = True
    stack_new_layers = True
    dropout_rate = 0.5
    n_classes = 18
    no_of_gpus = 1

    # Training Config
    n_splits = 7  # No of split for skfold cross validation
    batch_size = 32  # No of samples fit every step
    epochs = 50  # No of epochs
    lr = 0.005  # Optimizer learning rate

    # Read pre-generated dataset comprising of 3 columns (file, species, species_id)
    tDf = pd.read_csv(testDataset)

    # Load test images
    Xtest, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)

    # Training
    print("Start cross-validation training...")
    histories = []
    for iteration in range(n_splits):

        # Define model name
        model_name = '%s_model_{epoch:03d}_{val_acc:.2f}_iter%d.h5' % (model_type, iteration)
        filepath = os.path.join(model_dir, model_name)

        # Load model
        if no_of_gpus > 1:

            with tf.device("/cpu:0"):
                model = load_model(
                    model_type,
                    input_shape=input_shape + (3,),
                    n_classes=n_classes,
                    include_top=include_top,
                    stack_new_layers=stack_new_layers,
                    dropout_rate=dropout_rate
                )
            model = ModelMGPU(model, no_of_gpus)

        else:
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

        histories.append(history)

        K.clear_session()


if __name__ == "__main__":
    for model_type in ["Xception", "InceptionV3", "InceptionResNetV2", "ResNet50", "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge", "NASNetMobile"]:
        main(model_type)
