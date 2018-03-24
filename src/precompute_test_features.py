import glob
import numpy as np

import keras.backend as K
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
from tools.image_gen_extended import ImageDataGenerator
from tools.util import load_preprocess_input

K.clear_session()

modelroot = '../trained_models'
testpath = '../ttest'
batch_size=1024
all_models = {
    # 'InceptionResNetV2': [
    #     'iter0',
    #     'iter1',
    #     'iter2',
    #     'iter3',
    #     'iter4',
    #     'iter5',
    #     'iter6',
    # ],
    'InceptionV3': [
        'InceptionV3_model_025_0.86_iter0',
        'InceptionV3_model_025_0.87_iter1',
        'InceptionV3_model_003_0.83_iter2',
        'InceptionV3_model_014_0.86_iter3',
        'InceptionV3_model_004_0.83_iter4',
        'InceptionV3_model_018_0.87_iter5',
        'InceptionV3_model_019_0.87_iter6',
    ],
    'ResNet50': [
        'ResNet50_model_026_0.86_iter0',
        'ResNet50_model_004_0.82_iter1',
        'ResNet50_model_029_0.85_iter2',
        'ResNet50_model_005_0.81_iter3',
        'ResNet50_model_015_0.85_iter4',
        'ResNet50_model_005_0.81_iter5',
        'ResNet50_model_032_0.85_iter6',
    ],
    'Xception': [
        'Xception_model_018_0.86_iter0',
        'Xception_model_015_0.87_iter1',
        'Xception_model_021_0.87_iter2',
        'Xception_model_018_0.87_iter3',
        'Xception_model_016_0.87_iter4',
        'Xception_model_018_0.87_iter5',
        'Xception_model_013_0.86_iter6',
    ],
}

for modeltype, models in all_models.items():
    preprocess_input, input_shape = load_preprocess_input(modeltype)
    for val_id, modelpath in enumerate(models):
        print('init:', modeltype, modelpath, val_id)
        model = load_model('{}/{}.h5'.format(modelroot, modelpath))
        print('model loaded')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        model.compile('sgd', 'mse') # fix error Model' object has no attribute 'stateful_metric_names
        
        datagen = ImageDataGenerator()
        datagen.set_pipeline([lambda x, rng=None, **kwargs: preprocess_input(x),])

        print('init done')
        generator = datagen.flow_from_directory(
            directory=testpath,
            target_size=input_shape,
            class_mode=None,
            batch_size=batch_size,
        )

        X, y = None, None
        for Xnow in generator:
            start = (generator.batch_index - 1) * generator.batch_size
            ynow = generator.filenames[start:start+generator.batch_size]

            preds = model.predict(Xnow, verbose=1)
            
            if y == None:
                X = preds
                y = [fname[5:-5] for fname in ynow]
                print(y)
            else:
                X = np.append(X, preds, axis=0)
                y.extend([fname[5:-5] for fname in ynow])

            if X.shape[0] >= 16111:
                break

        # store
        mode = 'test'
        print('done {}-{}{}'.format(modeltype, mode, val_id))
        X, y = np.asarray(X), np.asarray(y)
        np.save('{}-{}{}.X'.format(modeltype, mode, val_id), X)
        np.save('{}-{}{}.y'.format(modeltype, mode, val_id), y)
        print('finish storing')
