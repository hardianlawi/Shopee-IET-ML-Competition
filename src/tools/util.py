import numpy as np
import keras.backend as K
from PIL import Image
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from keras.regularizers import l2
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import (
    vgg16,
    vgg19,
    inception_v3,
    resnet50,
    inception_resnet_v2,
    xception,
    densenet
)

def load_preprocess_input(model_type):

    if model_type == "VGG19":
        preprocess_input = vgg19.preprocess_input
        input_shape = (224, 224)
    elif model_type == "VGG16":
        preprocess_input = vgg16.preprocess_input
        input_shape = (224, 224)
    elif model_type == "InceptionResNetV2":
        preprocess_input = inception_resnet_v2.preprocess_input
        input_shape = (299, 299)
    elif model_type == "ResNet50":
        preprocess_input = resnet50.preprocess_input
        input_shape = (224, 224)
    elif model_type == "Xception":
        preprocess_input = xception.preprocess_input
        input_shape = (299, 299)
    elif "DenseNet" in model_type:
        preprocess_input = densenet.preprocess_input
        input_shape = (224, 224)
    else:
        preprocess_input = inception_v3.preprocess_input
        input_shape = (299, 299)

    return preprocess_input, input_shape


def load_model(model_type, input_shape, pooling=None, n_classes=None, include_top=True, stack_new_layers=True, flatten_fn=Flatten(), dropout_rate=0.5):
    # TODO: add params --> pooling, include_top
    # TODO: input shape
    if model_type == "VGG19":
        base_model = vgg19.VGG19(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)
    elif model_type == "VGG16":
        base_model = vgg16.VGG16(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)
    elif model_type == "InceptionResNetV2":
        base_model = inception_resnet_v2.InceptionResnetV2(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)
    elif model_type == "ResNet50":
        base_model = resnet50.ResNet50(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)
    elif model_type == "Xception":
        base_model = xception.Xception(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)
    else:
        base_model = inception_v3.InceptionV3(weights="imagenet", include_top=include_top, input_shape=input_shape, pooling=pooling)

    if include_top:

        # Pop last layer to fit class
        base_model.layers.pop()

        # Create last layer
        x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(base_model.layers[-1].output)

    else:

        x = base_model.output
        # x = flatten_fn(x)

        if stack_new_layers:

            x = AveragePooling2D(pool_size=(8, 8))(x)
            x = Dropout(.4)(x)
            x = Flatten()(x)
            x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(x)

    # Redefine model
    model = Model(inputs=base_model.input, outputs=x)

    return model


def load_images(filepaths, input_shape):

    all_imgs, broken_imgs = [], []

    for i, (j, filename) in enumerate(filepaths.iteritems()):
        if i % 1000 == 0:
            print('load', i)
        try:
            img = load_img(filename, target_size=input_shape)
            img = img_to_array(img).astype(np.float32)
            all_imgs.append(img)
        except:
            broken_imgs.append(j)
            # print(filename, "is broken")

    print('check', len(all_imgs), len(broken_imgs))
    return all_imgs, broken_imgs


def resize_image(img, thumbnail_size, to_array=True):
    tmp = Image.fromarray(img.astype(np.uint8))
    tmp = tmp.resize(thumbnail_size)
    if to_array:
        tmp = np.array(tmp).astype(np.float32)
    return tmp


class LearningRateTracker(Callback):

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        # If you want to apply decay.
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("At epoch %d" % iterations, "Learning rate is", K.eval(lr_with_decay))


def scheduler(epoch):
    if epoch < 15:
        return 0.01
    elif epoch < 28:
        return 0.001
    elif epoch < 40:
        return 0.0004
    elif epoch < 60:
        return 0.00008
    else:
        return 0.000009
