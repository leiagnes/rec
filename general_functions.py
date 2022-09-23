# Collection of functions that are used for all the models

import pandas as pd
import os
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score

# Folder with data files
# data_path = "C:/Users/agnes/OneDrive - Hochschule Luzern/REC_Assignment/"
data_path = "/Users/leo/switchdrive2/Private/REC/ILIAS/Project/"

# Folder and file name for evaluation overview file
# eval_path = "/Users/agnes/OneDrive - Hochschule Luzern/REC_Assignment/finalModels/output/"
eval_path = "/Users/leo/OneDrive - Hochschule Luzern/REC_Assignment/finalModels/output/"
eval_file = "model_comparison.csv"

# features for songs that are used for content-based methods
media_feature_definition = ['genre_id', 'artist_id', 'album_id']


# Function for reading the data from the file, preprocessing and train-test-splitting
def get_train_and_test_data(sample=False, return_media_features=False):
    # Reading data from file
    if sample:
        file = data_path + "train_sample.csv"
    else:
        file = data_path + "train.csv"
    full_dataset = pd.read_csv(file)

    # Preprocessing
    # The Matrix factorization needs IDs without gaps. Hence, the rank_ids are introduced.
    full_dataset['user_rank_id'] = full_dataset['user_id'].rank(method='dense')
    full_dataset['media_rank_id'] = full_dataset['media_id'].rank(method='dense')

    # Train-test-split
    train, test = train_test_split(full_dataset, test_size=0.2, random_state=1)

    # Preprocessing for only train-data
    # The data can have multiple entries per user-media-combination
    # Hence, a grouping is applied, taking the mean of the column 'is_listened' per combination
    train = train.groupby(['user_id', 'media_id', 'user_rank_id', 'media_rank_id'],
                          as_index=False)['is_listened'].mean()

    if return_media_features:
        media_features = full_dataset[['media_id'] + media_feature_definition].drop_duplicates()
        return train, test, media_features
    else:
        return train, test


# Function for calculating the model evaluation in the same way for all models based on the predictions.
# All the calculated predictions are then added to an overview file.
def do_evaluation(train, test, store=False, model_name=None):
    # general metrics
    evaluation = {'model_name': model_name,
                  'train_mse': mean_squared_error(y_true=train.is_listened, y_pred=train.prediction),
                  'test_mse': mean_squared_error(y_true=test.is_listened, y_pred=test.prediction),
                  'train_rmse': mean_squared_error(y_true=train.is_listened, y_pred=train.prediction, squared=False),
                  'test_rmse': mean_squared_error(y_true=test.is_listened, y_pred=test.prediction, squared=False),
                  'train_mae': mean_absolute_error(y_true=train.is_listened, y_pred=train.prediction),
                  'test_mae': mean_absolute_error(y_true=test.is_listened, y_pred=test.prediction)}
    pprint(evaluation)

    # ROC-curve and AUC-value
    fpr, tpr, _ = roc_curve(test.is_listened, test.prediction)
    auc = roc_auc_score(test.is_listened, test.prediction)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(eval_path + model_name + "_ROC.png")
    plt.close()
    evaluation['test_auc'] = auc

    if store:
        # store to file only when model_name is specified
        if model_name is None:
            raise ValueError("model_name has to be specified to print the evaluation to the file.")

        # Store evaluation to file
        file_path = eval_path + eval_file
        if os.path.exists(file_path):
            eval_df = pd.read_csv(file_path)
            eval_df = eval_df[eval_df.model_name != model_name]
        else:
            eval_df = pd.DataFrame()
        eval_df = pd.concat([eval_df, pd.DataFrame([evaluation])])
        eval_df.to_csv(file_path, index=False)
