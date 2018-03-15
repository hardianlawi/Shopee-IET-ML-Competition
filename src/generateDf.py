import argparse
import os
import glob
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', required=True)
args = parser.parse_args()

# dirpath = "/home/hardian_lawi/plant-seedlings-classification/datasets/train"
dirpath = args.dirpath

with open("../meta/mappings.txt", "r") as f:
    mappings = [l.strip().split()[0] for l in f.readlines()]
    mappings = dict(zip(mappings, range(len(mappings))))

print(mappings)

df = {}
df["file"] = []
df["category"] = []
for dirname in os.listdir(os.path.join(dirpath, "train")):
    for filename in glob.glob(os.path.join(dirpath, "train", dirname, "*")):
        df["file"].append(filename)
        df["category"].append(dirname)
df = pd.DataFrame(df)
df["category_id"] = df.category.map(mappings)
df[["file", "category", "category_id"]].to_csv(os.path.join(dirpath, "train.csv"), index=False)
