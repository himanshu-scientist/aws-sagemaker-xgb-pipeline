
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--train-output', type=str, required=True)
    parser.add_argument('--test-output', type=str, required=True)
    parser.add_argument('--label-column', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_data)

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df[args.label_column]
    )

    cols = [args.label_column] + [c for c in df.columns if c != args.label_column]
    train_df = train_df[cols]
    test_df = test_df[cols]

    train_df.to_csv(os.path.join(args.train_output, "train.csv"), index=False, header=False)
    test_df.to_csv(os.path.join(args.test_output, "test.csv"), index=False, header=False)
