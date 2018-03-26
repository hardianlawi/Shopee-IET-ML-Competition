import os
import glob
import numpy as np
import pandas as pd


def average_predictions(file_paths, save_to, for_submission=True):

    method = "AM"

    for i, path in enumerate(file_paths):
        if i == 0:
            df = pd.read_csv(path).drop("id", axis=1).as_matrix()
        else:
            if method == "AM":
                df += pd.read_csv(path).drop("id", axis=1).as_matrix()
            else:
                df *= pd.read_csv(path).drop("id", axis=1).as_matrix()

    if method == "AM":
        df /= (i+1)
    else:
        df = np.power(df, 1/(i+1))

    print(df)
    print(df.sum(axis=1))

    if for_submission:
        df = pd.concat([
            pd.DataFrame({
                "id": range(1, df.shape[0]+1),
                "category": np.argmax(df, axis=-1),
            }),
        ], axis=1)

        test_df = pd.read_csv("../outputs/submissions/xception.csv")

        print((df.category == test_df.category).mean())

    else:

        df = pd.concat([
            pd.DataFrame({"id": range(1, df.shape[0]+1)}),
            pd.DataFrame(df, columns=["f" + str(x) for x in range(df.shape[1])]),
        ], axis=1)

    df.to_csv(save_to, index=False)


def main():

    for model_type in ["InceptionResNetV2", "Xception", "ResNet50", "InceptionV3", "DenseNet201", "DenseNet169", "DenseNet121"]:
        file_paths = []
        save_to = "../outputs/test/avg/{}.csv".format(model_type)
        for_submission = False
        for path in glob.glob(os.path.join("../outputs/test", model_type, "*.csv")):
            file_paths.append(path)
            print(path)
        average_predictions(file_paths, save_to, for_submission=for_submission)


if __name__ == "__main__":
    main()
