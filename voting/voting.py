import collections
import glob
import numpy as np
import pandas as pd


ntests = 16111
dirs = [
    'InceptionV3',
    'InceptionResNetV2',
    'InceptionResNetV2',
    'Xception',
    'Xception',
    # 'ResNet50',
    'DenseNet201',
    'DenseNet121',
    'DenseNet169',
    'NASNetLarge',
    'NASNetMobile',
]

votes = []

for i in range(1, 4):
    for dirpath in dirs:
        for fpath in glob.glob('test/v{}/{}/*.csv'.format(i, dirpath)):
            df = pd.read_csv(fpath)
            df.drop('id', axis=1, inplace=True)
            if 'ann' in dirpath:
                votes.append(df.category)
            else:
                preds = np.argsort(df.as_matrix())[:, -1]
                votes.append(preds)

# voting part
print(np.array(votes).transpose())
votes = np.array(votes).transpose().tolist()
preds = [collections.Counter(vote).most_common(1)[0][0] for vote in votes]

df = pd.DataFrame()
df['id'] = [i + 1 for i in range(len(preds))]
df['category'] = preds
df.to_csv('votingall2.csv', index=False)
