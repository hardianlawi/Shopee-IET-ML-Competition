import gc
import os
import glob
import pandas as pd
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from tools.util import load_preprocess_input, load_default_input_shape, load_images


train_val_dir = "../data/train_val_v1"
testDataset = "../meta/mapTest.csv"

# Training Config
batch_size = 32  # No of samples fit every step
n_classes = 18

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
tDf = pd.read_csv(testDataset)

input_prev = None
for model_type in ["DenseNet121", 'DenseNet169', "DenseNet201", "NASNetMobile", "ResNet50", "InceptionResNetV2", "InceptionV3", "Xception", "NASNetLarge"]:

    if model_type == "NASNetLarge":
        batch_size = 16

    # Load preprocessor based on model_type
    preprocess_input = load_preprocess_input(model_type)

    input_shape = load_default_input_shape(model_type)

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
        cval=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocess_input,
        data_format=K.image_data_format()
    )

    # Load test images
    if input_shape != input_prev:
        Xtest, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)
        input_prev = input_shape

    # Convert and process test images
    Xtest = np.asarray(Xtest)
    Xtest = preprocess_input(Xtest)

    valDf = pd.DataFrame()

    val_dir = os.path.join("../outputs/val/v3", model_type)
    test_dir = os.path.join("../outputs/test/v3", model_type)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for f in sorted(glob.glob(os.path.join("../outputs/saved_models", model_type, "*")), key=lambda x: x[-4]):

        print("Model:", f)
        iteration = f[-4]

        model = load_model(f)

        print("Generating validation data...")
        val_generator = datagen.flow_from_directory(
            directory=os.path.join(train_val_dir, "train_val_%s" % iteration, "val"),
            target_size=input_shape,
            class_mode="categorical",
            batch_size=100000,
            seed=2018
        )

        for Xval, label in val_generator:
            val_predictions = model.predict(Xval)
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

        print("Generating prediction on test data...")
        test_predictions = model.predict(Xtest)
        testDf = pd.DataFrame({"id": tDf["id"]})
        testDf = pd.concat([testDf, pd.DataFrame(test_predictions, columns=[model_type+"_f"+str(x) for x in range(n_classes)])], axis=1)
        testDf.to_csv(
            os.path.join(test_dir, "%s_test_iter%s.csv" % (model_type, iteration)),
            index=False
        )

        del test_predictions, testDf
        gc.collect()

    valDf.to_csv(os.path.join(val_dir, "%s_val.csv" % model_type), index=False)
