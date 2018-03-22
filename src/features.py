import os
import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tools.util import load_preprocess_input


def train_generator(data, input_shape, batch_size):
    imgs = []
    categories = []
    fnames = []
    for i, (category, filename) in enumerate(data):
        if i % 1000 == 0:
            print('load', i)
        
        img = load_img(filename, target_size=input_shape)
        img = img_to_array(img).astype(np.float32)
        imgs.append(img)
        categories.append(category)
        fnames.append(filename)

        if len(imgs) == batch_size:
            yield imgs, categories, fnames
            imgs = []
            categories = []
            fnames = []
    
    if len(imgs) > 0:
        yield imgs, categories, fnames


def test_generator(data, input_shape, batch_size):
    imgs = []
    test_ids = []
    fnames = []
    for i, (test_id, filename) in enumerate(data):
        if i % 1000 == 0:
            print('load', i)
        
        img = load_img(filename, target_size=input_shape)
        img = img_to_array(img).astype(np.float32)
        imgs.append(img)
        test_ids.append(test_id)
        fnames.append(filename)

        if len(imgs) == batch_size:
            yield imgs, test_ids, fnames
            imgs = []
            test_ids = []
            fnames = []
    
    if len(imgs) > 0:
        yield imgs, test_ids, fnames


K.clear_session()

include_top = False
stack_new_layers = False
input_shape = (299, 299, 3)
batch_size = 2048
dropout_rate = 0.5

output_dir = "../outputs/training_v2"
logs_dir = "../logs"
dataset = "../mapTrain.csv"
testDataset = "../mapTest.csv"

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

models = [
    'InceptionV3_model_025_0.86_iter0',
    'InceptionV3_model_025_0.87_iter1',
    'InceptionV3_model_003_0.83_iter2',
    'InceptionV3_model_014_0.86_iter3',
    'InceptionV3_model_004_0.83_iter4',
    'InceptionV3_model_018_0.87_iter5',
    'InceptionV3_model_019_0.87_iter6',
]

for model_type in models:
    print("     Model %s" % model_type)
    preprocess_input, input_shape = load_preprocess_input(model_type)
    model = load_model('../trained_models/{}.h5'.format(model_type))
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    Xres, yres, fres = None, None, None
    for X, y, fs in train_generator(zip(df.category_id, df.file), input_shape, batch_size):
        X = preprocess_input(np.asarray(X))
        X = model.predict(X, verbose=1)
        
        if yres == None:
            Xres, yres, fres = X, y, fs
        else:
            Xres = np.append(Xres, X, axis=0)
            yres.extend(y)
            fres.extend(fs)
    
    # store
    trainDf = pd.DataFrame(Xres, columns=["f"+str(x) for x in range(Xres.shape[1])])
    trainDf["category_id"] = np.asarray(yres)
    trainDf["file"] = fres
    trainDf.to_csv("%s_train.csv" % (model_type), index=False)

    Xres, yres, fres = None, None, None
    for X, y, fs in test_generator(zip(tDf.id, tDf.file), input_shape, batch_size):
        X = preprocess_input(np.asarray(X))
        X = model.predict(X, verbose=1)
        
        if yres == None:
            Xres, yres, fres = X, y, fs
        else:
            Xres = np.append(Xres, X, axis=0)
            yres.extend(y)
            fres.extend(fs)
        
    # store
    testDf = pd.DataFrame()
    testDf["id"] = np.asarray(yres)
    testDf["file"] = fres
    testDf = pd.concat([testDf, pd.DataFrame(Xres, columns=["f"+str(x) for x in range(Xres.shape[1])])], axis=1)
    testDf.to_csv("%s_test.csv" % (model_type), index=False)
