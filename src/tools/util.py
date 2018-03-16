import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import (
    vgg16,
    vgg19,
    inception_v3,
    resnet50,
    inception_resnet_v2,
    xception
)


def load_preprocess_input(model_type):

    if model_type == "VGG19":
        preprocess_input = vgg19.preprocess_input
    elif model_type == "VGG16":
        preprocess_input = vgg16.preprocess_input
    elif model_type == "InceptionResNetV2":
        preprocess_input = inception_resnet_v2.preprocess_input
    elif model_type == "ResNet50":
        preprocess_input = resnet50.preprocess_input
    elif model_type == "Xception":
        preprocess_input = xception.preprocess_input
    else:
        preprocess_input = inception_v3.preprocess_input

    return preprocess_input


def load_model(model_type, input_shape, n_classes=None, include_top=True, stack_new_layers=True, flatten_fn=Flatten(), dropout_rate=0.5):

    if model_type == "VGG19":
        base_model = vgg19.VGG19(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "VGG16":
        base_model = vgg16.VGG16(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "InceptionResNetV2":
        base_model = inception_resnet_v2.InceptionResnetV2(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "ResNet50":
        base_model = resnet50.ResNet50(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "Xception":
        base_model = xception.Xception(weights="imagenet", include_top=include_top, input_shape=input_shape)
    else:
        base_model = inception_v3.InceptionV3(weights="imagenet", include_top=include_top, input_shape=input_shape)

    if include_top:

        # Pop last layer to fit class
        base_model.layers.pop()

        # Create last layer
        x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(base_model.layers[-1].output)

    else:

        x = base_model.output
        x = flatten_fn(x)

        if stack_new_layers:

            # x = Dense(4096, name="fc1")(x)
            # if dropout_rate:
            #     x = Dropout(dropout_rate)(x)
            # x = Dense(4096, name="fc2")(x)
            # if dropout_rate:
            #     x = Dropout(dropout_rate)(x)
            x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(x)

    # Redefine model
    model = Model(inputs=base_model.input, outputs=x)

    return model


def load_images(filepaths, input_shape):

    all_imgs, broken_imgs = [], []

    for i, filename in enumerate(filepaths):
        try:
            img = load_img(filename, target_size=input_shape)
            img = img_to_array(img).astype(np.float32)
            all_imgs.append(img)
        except:
            broken_imgs.append(i)
            print(filename, "is broken")

    return all_imgs, broken_imgs
