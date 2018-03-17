import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


train_dir = "../data/train"
output_dir = "../data/train_val"
n_splits = 7


def load_images(train_dir):

    all_images = []
    images_names = []
    categories = []

    for category in os.listdir(train_dir):
        for filename in os.listdir(os.path.join(train_dir, category)):
            try:
                img = plt.imread(os.path.join(train_dir, category, filename))
                all_images.append(img)
                images_names.append(filename)
                categories.append(category)
            except:
                print(filename, "is broken.")

    return all_images, images_names, categories


def save_images(dir_path, all_images, images_names, categories, iteration, dtype):

    for img, filename, category in zip(all_images, images_names, categories):

        if dtype == "train":

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "train", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "train", category))

            plt.imsave(
                fname=os.path.join(dir_path, "train_val_%d" % iteration, "train", category, filename),
                arr=img,
            )

        else:

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "val", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "val", category))

            plt.imsave(
                fname=os.path.join(dir_path, "train_val_%d" % iteration, "val", category, filename),
                arr=img,
            )


all_images, images_names, categories = load_images(train_dir)

# Define a splitter
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)

for i, (train, val) in enumerate(skf.split(all_images, categories)):

    save_images(
        dir_path=output_dir,
        all_images=np.array(all_images)[train],
        images_names=np.array(images_names)[train],
        categories=np.array(categories)[train],
        iteration=i,
        dtype="train"
    )

    save_images(
        dir_path=output_dir,
        all_images=np.array(all_images)[val],
        images_names=np.array(images_names)[val],
        categories=np.array(categories)[val],
        iteration=i,
        dtype="val"
    )
