# Model fitting: Matrix factorization

import general_functions

import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model

use_sample = False
model_name = "Matrix_factorization"
if use_sample:
    model_name = model_name + "_sample"


def main():
    # get the data
    train, test = general_functions.get_train_and_test_data(sample=use_sample)

    # derive the number of unique users and songs
    full_data = pd.concat([train, test])
    n_users = full_data['user_rank_id'].nunique()
    n_media = full_data['media_rank_id'].nunique()

    # creating media embedding path
    media_input = Input(shape=[1], name="Media-Input")
    media_embedding = Embedding(n_media + 1, 5, name="Media-Embedding")(media_input)
    media_vec = Flatten(name="Flatten-Media")(media_embedding)

    # creating user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    # performing dot product and creating model
    prod = Dot(name="Dot-Product", axes=1)([media_vec, user_vec])
    model = Model([user_input, media_input], prod)
    model.compile('adam', 'mean_squared_error')

    # fitting the model and showing the fitting progress.
    history = model.fit([train.user_rank_id, train.media_rank_id], train.is_listened, epochs=15, verbose=1)
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.savefig(f"output/{model_name}_fitting.png")
    plt.close()

    # predict the train and test data in order to perform the model evaluation
    train['prediction'] = model.predict([train.user_rank_id, train.media_rank_id])
    test['prediction'] = model.predict([test.user_rank_id, test.media_rank_id])

    general_functions.do_evaluation(train=train, test=test, store=True, model_name=model_name)


if __name__ == '__main__':
    main()
