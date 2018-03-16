import os
import numpy as np
import pandas as pd

import keras.backend as K
from tools.util import load_model, load_images, load_preprocess_input

K.clear_session()

# Configuration
include_top = False
stack_new_layers = False
input_shape = (299, 299, 3)
dropout_rate = 0.5

# Filepaths
output_dir = "../outputs/training_v2"
logs_dir = "../logs"
dataset = "../data/mapTrain.csv"
testDataset = "../data/mapTest.csv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_dir = os.path.join(output_dir, "train")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

test_dir = os.path.join(output_dir, "test")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
df = pd.read_csv(dataset).sample(frac=1.0, random_state=2018)
tDf = pd.read_csv(testDataset)

# Load and resize all images
print("Loading images...")
all_imgs, broken_imgs = load_images(df.file, input_shape=input_shape)
test_imgs, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)

# Target label
y = df.drop(broken_imgs).as_matrix(columns=["category_id"])  # Convert target to numpy array of m x 1

# Start Generating features
print("Start generating features...")
for model_type in ["InceptionV3", "VGG16", "VGG19", "ResNet50", "Xception", "InceptionResnetV2"]:

    print("     Model %s" % model_type)

    model = load_model(
        model_type=model_type,
        input_shape=input_shape,
        include_top=include_top,
        stack_new_layers=stack_new_layers
    )

    preprocess_input = load_preprocess_input(model_type)

    # Convert and process train images to numpy array
    X = preprocess_input(np.array(all_imgs))  # Matrix of (m x input_shape)
    Xtest = preprocess_input(np.array(test_imgs))

    # Generate features

    #    Training
    print("     Training...")
    X = model.predict(X)
    trainDf = pd.DataFrame(X, columns=["f"+str(x) for x in range(X.shape[1])])
    trainDf["category_id"] = y
    trainDf.to_csv("%s_train.csv" % model_type, index=False)

    #   Test
    print("     Test...")
    Xtest = model.predict(Xtest)
    testDf = pd.DataFrame()
    testDf["id"] = tDf["id"]
    testDf = pd.concat([testDf, pd.DataFrame(Xtest, columns=["f"+str(x) for x in range(X.shape[1])])], axis=1)
    testDf.to_csv("%s_test.csv" % model_type, index=False)
