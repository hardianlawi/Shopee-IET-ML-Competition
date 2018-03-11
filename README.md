# Shopee-IET-ML-Competition

The aim of the project is to build a model that can predict the classification of the input images. The training/test sets are images provided by Shopee, which are classified into 18 categories.

# Steps

## Data Exploration

In this step, we randomly visualize a sample of images and get rough idea of how the data are distributed.

Some things to try:

- Number of images in each group
- Number of images across different sizes.
- Images with watermark / weird logos.

## Data Preprocessing

In this step, we create a pipeline of preprocessing steps to generate a final dataset of images with equal sizes.

P.S. Size of an image will depend on the neural network used (I usually use the default size from the paper, but this could change depending on the dataset).

### Data Augmentation

This step is only to generate additional datasets. Things to try:

- Mirroring
- Random Cropping (Crop a large portion of image, probably there are some common heuristic for this)
- Rotation
- Shearing
- Color shifting (See PCA Color Augmentation for more advanced algorithms)

Apply this only if it makes sense to do so. Common implementation for this is to use some CPU threads to perform data augmentation s.t. some processes load data from harddisk, some processes perform data augmentation, and some other processsed perform Training (GPU). I think people normally use [`ImageDataGenerator`](https://keras.io/preprocessing/image/) for `keras` or manually assigning CPU and GPU for `tensorflow`.

### Resize images

Do not forget to resize the original images to the size you want to fit to your model.

### Public Dataset

If there are public datasets that have the same distribution as our test data, please recommend.

## Modelling

Things to try:

- All pretrained model available in [`keras` API](https://keras.io/applications/). For starters, try fine-tuning only the last few layers of the pre-trained model (Do not forget to freeze all the layers you do not want to fine tune). Also, replace the final layer to suit the number of classes of the problem which in this case is 18.
- Simple Conv Neural Network with different hyperparameters.
- Other pretrained models that have been trained for this particular problem.
- Retrain the whole pretrained models (Be cautious of overfitting, because our dataset doesn't seem to be that large.

Do not forget to always save the model trained with details unless you only want to do some quick dirty prototyping.

## Evaluation

This step has to be synchronized with the `Modelling` part and the part that we have to fix before working on our solutions.

Different evaluation methods that are possible:
- K-Fold Cross Validation / Stratified K-Fold (More robust if the dataset is small, but more expensive)
- Holdout / Stratified Holdout (Good if the dataset is enough)

## Submission

Generate submission with the same format as `sample submission.csv`. Always save with different name files that explains how they are generated.


# Attempts Made
