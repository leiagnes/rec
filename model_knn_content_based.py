# Model fitting: Content-based (item-based) Filtering with k-nearest neighbors

import general_functions

from collections import Counter

use_sample = False
model_name = "knn_content_based"
if use_sample:
    model_name = model_name + "_sample"

prediction_count = 0


def main():
    # get the data
    train, test, media_features = general_functions.get_train_and_test_data(sample=use_sample,
                                                                            return_media_features=True)

    # Computing music popularity ranks
    no_of_ratings = dict(Counter(train['media_id']).items())
    rankings = {}
    rank = 1
    for musicID, ratingCount in sorted(no_of_ratings.items(), key=lambda x: x[1], reverse=True):
        rankings[musicID] = rank
        rank += 1
    media_features['rank'] = media_features['media_id'].map(rankings)
    media_features.sort_values(by='rank', inplace=True)

    # predict the train and test data in order to perform the model evaluation
    # (No fitting needed, all logic is in the predict function)
    test['prediction'] = test.apply(lambda x: predict(media_id=x['media_id'],
                                                      user_id=x['user_id'],
                                                      k=5,
                                                      train=train,
                                                      media_features=media_features), axis=1)
    train['prediction'] = train.apply(lambda x: predict(media_id=x['media_id'],
                                                        user_id=x['user_id'],
                                                        k=5,
                                                        train=train,
                                                        media_features=media_features), axis=1)

    general_functions.do_evaluation(train=train,
                                    test=test,
                                    store=True,
                                    model_name=model_name)


def predict(media_id, user_id, k, train, media_features):
    global prediction_count
    prediction_count += 1
    if prediction_count % 1000 == 0:
        print(f"Prediction number {prediction_count}")
    # first select only songs, that were rated by the user,
    # because it makes only sense to select these as nearest neighbors.
    subset_media_ids = train[train.user_id == user_id].media_id.unique()
    media_subset = media_features[media_features.media_id.isin(subset_media_ids)]

    # derive the attributes of the song to predict
    genre, artist, album = \
        media_features[media_features.media_id == media_id][['genre_id', 'artist_id', 'album_id']].values.tolist()[0]

    # phase 1: search for songs where all 3 attributes are matching
    nearest_neighbors = set(media_subset[(media_subset.genre_id == genre) &
                                         (media_subset.artist_id == artist) &
                                         (media_subset.album_id == album)].head(k).media_id)

    if len(nearest_neighbors) < k:
        # phase 2: search for songs where 2 attributes are matching
        nearest_neighbors.update(media_subset[((media_subset.genre_id == genre) & (media_subset.artist_id == artist)) |
                                              ((media_subset.genre_id == genre) & (media_subset.album_id == album)) |
                                              ((media_subset.album_id == album) & (media_subset.artist_id == artist))
                                              ].head(k - len(nearest_neighbors)).media_id)

        if len(nearest_neighbors) < k:
            # phase 3: search for songs where only one attribute is matching
            nearest_neighbors.update(
                media_subset[(media_subset.genre_id == genre) |
                             (media_subset.artist_id == artist) |
                             (media_subset.album_id == album)
                             ].head(k - len(nearest_neighbors)).media_id)

    if len(nearest_neighbors) == 0:
        print(
            f"{prediction_count}: No neighbors found for user {user_id} and item {media_id}. Using neutral prediction.")
        return 0.5

    # return rating average from nearest neighbors
    return train[(train.user_id == user_id) & (train.media_id.isin(nearest_neighbors))].is_listened.mean()


if __name__ == '__main__':
    main()
