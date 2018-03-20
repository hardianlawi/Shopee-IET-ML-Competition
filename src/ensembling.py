import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tools.util import generate_data, NNModel, LearningRateTracker, scheduler


train_files = [
    "../outputs/val/InceptionV3_val.csv",
    "../outputs/val/Xception_val.csv",
]

test_files = [
    "../outputs/test/InceptionV3_test.csv",
    "../outputs/test/Xception_test.csv",
]


n_splits = 7


train = generate_data(train_files, ylabel="category_id")
test = generate_data(test_files, mode="test")

params = {
    'subsample': 0.8, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8,
    'gamma': 0, 'min_child_weight': 1, 'reg_alpha': 0, 'reg_lambda': 1,
    'max_depth': 4, 'learning_rate': 0.05, 'objective': 'multi:softmax',
    'seed': 0, 'eval_metric': ['merror'], "num_class": 18
}

X = train.drop("category_id", axis=1)
y = train.category_id

sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9)
model = NNModel(X.shape[1], 18)
model.compile(sgd, "categorical_crossentropy", metrics=["accuracy"])

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)
for train, val in skf.split(X, y):
    Xtrain = X[train, :]
    ytrain = y[train, :]
    Xval = X[val, :]
    yval = y[val, :]

    # Prepare callbacks
    lr_reducer = ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.2,
        cooldown=0,
        patience=1,
        min_lr=0.5e-6
    )

    early_stopping = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=4,
        verbose=0,
        mode='auto'
    )

    lr_scheduler = LearningRateScheduler(scheduler)

    lr_tracker = LearningRateTracker()

    callbacks = [lr_reducer, early_stopping, lr_scheduler, lr_tracker]

    history = model.fit(x=Xtrain, y=ytrain, batch_size=64, epochs=50)


dtrain = xgb.DMatrix(X, label=y)
res = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=n_splits,
    seed=2018,
    stratified=True,
    early_stopping_rounds=10)

clf = xgb.train(params, dtrain, num_boost_round=42, verbose_eval=True)

Xtest = test.drop("id", axis=1)
dtest = xgb.DMatrix(test.drop("id", axis=1))
ypreds = clf.predict(dtest)

submission_df = test[["id"]]
submission_df["category"] = ypreds.astype(int)
