import glob
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tools.util import load_preprocess_input

K.clear_session()

root = '..'
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
vals = ['train_val_{}'.format(i) for i in range(7)]

for modeltype, models in all_models.items():
    preprocess_input, input_shape = load_preprocess_input(modeltype)
    for modelpath, (val_id, valpath) in zip(models, enumerate(vals)):
        print('init:', modeltype, modelpath, val_id, valpath)
        model = load_model('{}/trained_models/{}.h5'.format(root, modelpath))
        print('model loaded')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        model.compile('sgd', 'mse') # fix error Model' object has no attribute 'stateful_metric_names
        
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

        print('init done')
        for mode in ['train', 'val']:
            print(modeltype, modelpath, val_id, valpath, mode)
            generator = datagen.flow_from_directory(
                directory='{}/train_val_v1/{}/{}/'.format(root, valpath, mode),
                target_size=input_shape,
                class_mode="categorical",
                batch_size=batch_size,
                seed=2018
            )

            X, y = None, None
            for Xnow, ynow in generator:
                preds = model.predict(Xnow, verbose=1)
                if y == None:
                    X, y = preds, ynow.tolist()
                else:
                    X = np.append(X, preds, axis=0)
                    y.extend(ynow.tolist())
                
                print('{} done for {}-{}{}'.format(len(y), modeltype, mode, val_id))
                if len(y) % 1024 != 0:
                    break
            
            # store
            print('done {}-{}{}'.format(modeltype, mode, val_id))
            X, y = np.asarray(X), np.asarray(y)
            np.save('{}-{}{}.X'.format(modeltype, mode, val_id), X)
            np.save('{}-{}{}.y'.format(modeltype, mode, val_id), y)
            print('finish storing')
