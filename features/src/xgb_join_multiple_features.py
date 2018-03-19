import numpy as np
import sys
import time
import xgboost

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from utils import load_data

n_thread = 16
n_splits = 7
num_round = 100
eta = 0.01
gamma = 0.4
max_depth = 8
min_child_weight = 3
random_state = 2018

fnames = [
    'InceptionV3_model_025_0.86_iter0_train',
    'InceptionV3_model_025_0.87_iter1_train',
    'InceptionV3_model_003_0.83_iter2_train',
    'InceptionV3_model_014_0.86_iter3_train',
    'InceptionV3_model_004_0.83_iter4_train',
    'InceptionV3_model_018_0.87_iter5_train',
    'InceptionV3_model_019_0.87_iter6_train',
]

X, y = None, None
skfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
train_val_generator = None
for fold, fname in enumerate(fnames):
    Xfold, yfold = load_data(fname)

    if train_val_generator == None:
        train_val_generator = skfold.split(Xfold, yfold)

    train_idx, test_idx = next(train_val_generator)
    if y == None:
        X = Xfold[test_idx]
        y = yfold[test_idx].tolist()
    else:
        X = np.append(X, Xfold[test_idx], axis=0)
        y.extend(yfold[test_idx].tolist())

print('finish generating new train data')
# train
y = np.asarray(y)

params = {
    'objective': 'multi:softprob',
    'num_class': len(set(y)),
    'eta': eta,
    'gamma': gamma,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0.01,
    'nthread': n_thread,
}

fname = 'InceptionV3_model_fold_all'
dtrain = xgboost.DMatrix(X, label=y)
model = xgboost.train(params, dtrain, num_boost_round=num_round, verbose_eval=True)
print('fold-{} total time {} s'.format('all', time.time() - start_train_time))
model.save_model('dmatrix2/{}-round{}-gamma{}-depth{}-mcw{}.model'.format(
                            fname, num_round, gamma, max_depth, min_child_weight))

"""
for fold, (train_idx, test_idx) in enumerate(skfold.split(Xfold, yfold)):
    fname = 'InceptionV3_model_fold{}'.format(fold)
    dtrain = xgboost.DMatrix(X[train_idx], label=y[train_idx])
    dtest = xgboost.DMatrix(X[test_idx], label=y[test_idx])
    print('fold', fold, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), file=sys.stderr)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(set(y)),
        'eta': eta,
        'gamma': gamma,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'lambda': 1,
        'alpha': 0.01,
        'nthread': n_thread,
    }
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    # train
    start_train_time = time.time()
    model = xgboost.train(params, dtrain, num_boost_round=num_round, evals=watchlist, 
                                    early_stopping_rounds=5, verbose_eval=True)
    print('fold-{} total time {} s'.format(fold, time.time() - start_train_time))

    # test
    preds = np.asarray([np.argsort(pred)[-1] for pred in model.predict(dtest)])
    y_test = y[test_idx]

    acc = accuracy_score(y_test, preds)
    print("{} fold-{} acc={}".format(fname, fold, acc))
    
    model.save_model('dmatrix2/{}-round{}-gamma{}-depth{}-mcw{}-{:.2f}.model'.format(
                                fname, num_round, gamma, max_depth, min_child_weight, acc))
"""