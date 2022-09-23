# Model fitting: Singular value decomposition

import general_functions

import pandas as pd
from surprise import SVD, Reader, Dataset

use_sample = False
model_name = "svd"
if use_sample:
    model_name = model_name + "_sample"


def main():
    # get the data
    train, test = general_functions.get_train_and_test_data(sample=use_sample)

    # convert the datasets to a surprise train-set and to correct format for predictions
    reader = Reader(rating_scale=(0, 1))
    surprise_train = Dataset.load_from_df(train[["user_id", "media_id", "is_listened"]], reader).build_full_trainset()
    list_train = train[["user_id", "media_id", "is_listened"]].to_numpy().tolist()
    list_test = test[["user_id", "media_id", "is_listened"]].to_numpy().tolist()

    # fit the model
    user_knn = SVD(random_state=10)
    user_knn.fit(surprise_train)

    # predict the train and test data in order to perform the model evaluation
    train_predictions = user_knn.test(list_train)
    test_predictions = user_knn.test(list_test)
    # the predicted values are added to the original dataframes
    train = train.reset_index(drop=True).join(pd.DataFrame(train_predictions)['est'])
    test = test.reset_index(drop=True).join(pd.DataFrame(test_predictions)['est'])

    general_functions.do_evaluation(train=train.rename(columns={'est': 'prediction'}),
                                    test=test.rename(columns={'est': 'prediction'}),
                                    store=True,
                                    model_name=model_name)


if __name__ == '__main__':
    main()
