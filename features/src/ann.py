import json
import numpy as np
import sys
import time

from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
from utils import load_data


def get_best(arr):
    if len(arr) == 1:
        return arr[0]

    cnt = {}
    for pos, x in enumerate(arr):
        if x not in cnt:
            cnt[x] = [1, pos]
        else:
            cnt[x][0] += 1

    if len(cnt) == 1:
        for k in cnt:
            return k

    res = sorted(((v, k) for k, v in cnt.items()), reverse=True)
    if res[0][0][0] > res[1][0][0]:
        return res[0][1]

    res = sorted(((pos, cat) for (count, pos), cat in res if res[0][0][0] == count))
    return res[0][1]


# constants
fname = 'ResNet50-train6'
val_fname = 'ResNet50-val6'
n_trees = 50
n_nearest = 5
random_state = 2018

# load data
X_train, y_train = load_data(fname)
y_train = np.argmax(y_train, axis=1)
n_categories = len(set(y_train))

# split train test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_test, y_test = load_data(val_fname)
y_test = np.argmax(y_test, axis=1)
print('train={}, test={}'.format(len(X_train), len(X_test)), file=sys.stderr)

# ANN
n_features = X_train[0].shape[0]
tree = AnnoyIndex(n_features)

start_train_time = time.time()
for idx, feature_vector in enumerate(X_train):    
    tree.add_item(idx, feature_vector)

print('build tree', file=sys.stderr)
tree.build(n_trees)
print('finished training, {} s'.format(time.time() - start_train_time), file=sys.stderr)

# store
tree.save('annoy/{}.ann'.format(fname))
np.save('annoy/{}.ytrain'.format(fname), y_train)

# test
acc = 0
for idx, (feature_vector, category) in enumerate(zip(X_test, y_test)):
    nn = tree.get_nns_by_vector(feature_vector, n_nearest)
    preds = [y_train[x] for x in nn]
    pred = get_best(preds)

    if idx % 1000 == 0:
        print('test {}/{}, pred={}, cat={}'.format(idx, len(X_test), pred, category), file=sys.stderr)
    if pred == category:
        acc += 1

print(acc, len(X_test))
print('acc=', acc/len(X_test))
