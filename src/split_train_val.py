import os
import argparse
from shutil import copyfile
from sklearn.model_selection import StratifiedKFold


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default="../data/train")
parser.add_argument("--output_dir", default="../data/train_val_v1")
parser.add_argument("--n_splits", default=7, type=int)
args = parser.parse_args()

train_dir = args.train_dir
output_dir = args.output_dir
n_splits = args.n_splits


def load_images(train_dir):
    print('split train dir', train_dir)
    all_images = []
    images_names = []
    categories = []

    for category in os.listdir(train_dir):
        for filename in os.listdir(os.path.join(train_dir, category)):
            try:
                all_images.append(os.path.join(train_dir, category, filename))
                images_names.append(filename)
                categories.append(category)
            except:
                print(filename, "is broken.")

    return all_images, images_names, categories


def save_images(dir_path, all_images, images_names, categories, iteration, dtype):
    print(save_images, dir_path, len(all_images), iteration)
    for img, filename, category in zip(all_images, images_names, categories):

        if dtype == "train":

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "train", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "train", category))
            copyfile(img, os.path.join(dir_path, "train_val_%d" % iteration, "train", category, filename))

        else:

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "val", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "val", category))
            copyfile(img, os.path.join(dir_path, "train_val_%d" % iteration, "val", category, filename))


all_images_dir, images_names, categories = load_images(train_dir)
print('loaded')

# Define a splitter
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)
print('defined')
for ii, (train, val) in enumerate(skf.split(all_images_dir, categories)):

    print('iteration', ii, train, val)

    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i in train:
        tmp1.append(all_images_dir[i])
        tmp2.append(images_names[i])
        tmp3.append(categories[i])

    tmp11 = []
    tmp22 = []
    tmp33 = []
    for i in val:
        tmp11.append(all_images_dir[i])
        tmp22.append(images_names[i])
        tmp33.append(categories[i])

    save_images(
        dir_path=output_dir,
        all_images=tmp1,
        images_names=tmp2,
        categories=tmp3,
        iteration=ii,
        dtype="train"
    )

    save_images(
        dir_path=output_dir,
        all_images=tmp11,
        images_names=tmp22,
        categories=tmp33,
        iteration=ii,
        dtype="val"
    )
