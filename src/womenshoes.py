import time
import numpy as np
import keras.backend as K

from tools import image_gen_extended as T
from PIL import Image


datagen = T.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
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
)

crop_datagen = T.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
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
)
crop_datagen.config['center_crop_size'] = (299, 299)
crop_datagen.set_pipeline([T.center_crop])

zoom_datagen = T.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.5,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
)

generator = datagen.flow_from_directory(
    directory='../train_womenshoes',
    target_size=(299, 299),
    class_mode='categorical',
    batch_size=1,
    seed=2018,
)

crop_generator = crop_datagen.flow_from_directory(
    directory='../train_womenshoes',
    target_size=(399, 399),
    class_mode='categorical',
    batch_size=1,
    seed=2018,
)

zoom_generator = zoom_datagen.flow_from_directory(
    directory='../train_womenshoes',
    target_size=(299, 299),
    class_mode='categorical',
    batch_size=1,
    seed=2018,
)

for (Xraw, yraw), (Xcrop, ycrop), (Xzoom, yzoom) in zip(generator, crop_generator, zoom_generator):
    img = Image.fromarray(Xraw[0].astype(np.uint8))
    img.show()
    time.sleep(0.2)

    img = Image.fromarray(Xcrop[0].astype(np.uint8))
    img.show()
    time.sleep(0.2)

    img = Image.fromarray(Xzoom[0].astype(np.uint8))
    img.show()
    input() # block
