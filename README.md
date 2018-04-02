# [Shopee-IET-ML-Competition](https://www.kaggle.com/c/shopee-iet-machine-learning-competition)

The aim of the project is to build a model to classify images to 18 categories. The training/test sets are images provided by Shopee, which are classified into 18 categories.

Metrics: `Accuracy`

## Getting Started

These instructions will get you a copy of the project up and running on your machine for development and testing purposes.

### Prerequisites

First, you have to download the data from [here](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) and store them under `data` folder in the folder of this repository. Furthermore, The script has been only tested using *Python 3.6* and for the libraries needed to run the script, you can check `requirements.txt`.

### Notebooks

The notebook contains the analysis done before and after the training.

- `EDA.ipynb`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/EDA.ipynb]: Exploratory data analysis, the analysis done before training.
- `Model Analysis.ipynb`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/Model%20Analysis.ipynb]: Analysis of the error of the models and correlation between models, done after the training.

### Scripts

Below are the details of the scripts:

- `split_train_val.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/split_train_val.py]: to split the `train` images to `n` folds and store them in a directory.
- `training.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/training.py]: to train the model
- `generate_df.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/generate_df.py]: to generate `mapTest.csv` which contain a sorted filepath to the test data.
- `generate_test.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/generate_test.py]: to generate `validation` and `test` predictions from the models trained.
- `average.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/average.py]: to average the predictions of `validation` or `test`.
- `voting.py`[https://github.com/hardianlawi/Shopee-IET-ML-Competition/blob/master/src/voting.py]: to do a weighted majority voting

### Training

First, you will have to run `split_train_val.py` to create a locally stored splits `train` data.

```
python split_train_val.py --train_dir ../data/train --output_dir ../data/train_val_v1 --n_splits 7
```

Next, you can immediately do the training by running the script `training.py`. All the configs can be easily configured from inside the script.

**P.S.** It is recommended to run the script from inside the folder to avoid `RelativePathError`, or equivalently do `cd src`.

## Result

After fine-tuning several pretrained models combined with weighted majorith voting, we achieved *0.86956* in the private leaderboard.

## Authors

Team: *h1n4*

* **Hardian Lawi** - [hardianlawi](https://github.com/hardianlawi)
* **Kevin Luvian** - [kevinluvian](https://github.com/kevinluvian)
* **Timothy Gabriel Kurniawan** - [timothygk](https://github.com/timothygk)

## Acknowledgments

Extending my gratitude to *NTU-IET* and *Shopee* who made this competition possible.
