import numpy as np
import pandas as pd
import sys
import time
import xgboost

from utils import load_data

fnames = [
    'InceptionV3_model_003_0.83_iter2_test',
    'InceptionV3_model_004_0.83_iter4_test',
    'InceptionV3_model_014_0.86_iter3_test',
    'InceptionV3_model_018_0.87_iter5_test',
    'InceptionV3_model_019_0.87_iter6_test',
    'InceptionV3_model_025_0.86_iter0_test',
    'InceptionV3_model_025_0.87_iter1_test',
]

n_category = 18

model_name = 'InceptionV3_model_019_0.87_iter6_train-round100-gamma0.4-depth8-mcw3-0.94'

model = xgboost.Booster({'nthread': 16})
model.load_model('dmatrix2-old/{}.model'.format(model_name))

ans = []
for fname in fnames:
    Xnow, testids = load_data(fname)
    
    X = np.asarray([feature for idx, feature in sorted(zip(testids, Xnow))])
    dtest = xgboost.DMatrix(X)
    preds = [np.argsort(pred)[-1] for pred in model.predict(dtest)]
    ans.append(preds)

# voting
print(len(ans), len(ans[0]))
preds = []
for i in range(len(ans[0])):
    counter = [0] * n_category
    for j in range(len(fnames)):
        counter[ans[j][i]] += 1
    
    preds.append(sorted(((v, k) for k, v in enumerate(counter)))[-1][1] + 1)

df = pd.DataFrame()
df['id'] = [i + 1 for i in range(len(preds))]
df['category'] = preds
df.to_csv('predict/{}.csv'.format(model_name), index=False)
