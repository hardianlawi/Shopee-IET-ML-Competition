import glob
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tools.util import generate_data

save_to = "../outputs/submissions/xgb_ensemble.csv"

train_files = []
for path in glob.glob("../outputs/val/v2/*"):
    train_files.append(path)

# train_files = [
#     "../outputs/val/InceptionV3_val.csv",
#     "../outputs/val/Xception_val.csv",
# ]

test_files = []
for path in glob.glob("../outputs/test/avg/v2/*"):
    test_files.append(path)

# test_files = [
#     "../outputs/test/InceptionV3_test.csv",
#     "../outputs/test/Xception_test.csv",
# ]

n_splits = 7

train = generate_data(train_files, label="category_id")
test = generate_data(test_files, mode="test")

params = {
    'subsample': 1.0, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8,
    'gamma': 0.1, 'min_child_weight': 2, 'reg_alpha': 0, 'reg_lambda': 1,
    'max_depth': 5, 'learning_rate': 0.001, 'objective': 'multi:softmax',
    'seed': 0, 'eval_metric': ['merror'], "num_class": 18, "silent": 1
}

X = train.drop("category_id", axis=1)
y = train.category_id

kf = KFold(n_splits=n_splits)
accs = []
for i, (train, val) in enumerate(kf.split(X, y)):
    dtrain = xgb.DMatrix(X.iloc[train].as_matrix(), label=y[train])
    clf = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    dtest = xgb.DMatrix(X.iloc[val].as_matrix(), label=y[val])
    ypreds = clf.predict(dtest)
    accs.append(accuracy_score(y[val], ypreds))
    print("Accuracy iter %d:" % i, accuracy_score(y[val], ypreds), train, val)

print("Average accuracy:", np.mean(accs))
# dtrain = xgb.DMatrix(X.as_matrix(), label=y)
# res = xgb.cv(
#     params,
#     dtrain,
#     num_boost_round=100,
#     nfold=n_splits,
#     seed=2018,
#     stratified=True,
#     early_stopping_rounds=10
# )

clf = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

Xtest = test.drop("id", axis=1)
dtest = xgb.DMatrix(test.drop("id", axis=1).as_matrix())
ypreds = clf.predict(dtest)

submission_df = test[["id"]]
submission_df["category"] = ypreds.astype(int)
submission_df.to_csv(save_to, index=False)
