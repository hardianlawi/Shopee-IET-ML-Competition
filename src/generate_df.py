import argparse
import os
import pandas as pd


def main():

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument("--meta_path", default="../meta")
    args = parser.parse_args()

    # Dir paths
    test_path = args.test_path
    meta_path = args.meta_path

    # Create a dataframe
    df = {}
    df["file"] = [os.path.join(test_path, f) for f in sorted(test_path, key=lambda x: int(x[5:-4]))]
    df["id"] = range(os.listdir(test_path))
    df = pd.DataFrame(df)

    df[["id", "file"]].to_csv(os.path.join(meta_path, "mapTest.csv"), index=False)


if __name__ == "__main__":
    main()
