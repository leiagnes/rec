# Creates a sample of the train data for faster developing
# Please note that the developed code should in the end be executed for the whole data and not only for the sample.
# to keep the same level of sparsity, the data is filtered by both, users and items

import numpy as np
import pandas as pd

# Pandas options to ensure proper printing to console
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Folder with data files
data_path = "/Users/leo/switchdrive2/Private/REC/ILIAS/Project/"

# Percentage of users and items to be included in the sample
item_user_prc = 0.2

# Ensure reproducibility
np.random.seed(0)


def main():
    # Reading the full dataset
    df = pd.read_csv(data_path + "train.csv")

    # Descriptive statistics to compare the original dataframe with the sample
    print(df.describe())

    # randomly select a sample of the user_ids/media_ids
    unique_media_ids = df['media_id'].unique()
    unique_user_ids = df['user_id'].unique()
    sample_media_ids = np.random.choice(unique_media_ids, size=int(len(unique_media_ids)*item_user_prc), replace=False)
    sample_user_ids = np.random.choice(unique_user_ids, size=int(len(unique_user_ids) * item_user_prc), replace=False)

    # Filter the dataset according to the selected ids
    sample = df[(df.user_id.isin(sample_user_ids)) & (df.media_id.isin(sample_media_ids))]

    # Descriptive statistics to compare the original dataframe with the sample
    print(sample.describe())

    # Save as csv file
    sample.to_csv(data_path + "train_sample.csv", index=False)
    print(f"Saved sample data with {len(sample)} records.")


if __name__ == '__main__':
    main()
