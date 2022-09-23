# Music Recommendation System
Authors: Agnes Lei, Leo Rettich, Wellhöfer Patricia <br><br>
Completed in March 2022<br><br>
<b>Dataset</b>:<br>
- freely accessible on https://www.kaggle.com/c/dsg17-online-phase/overview
- from Deezer, a French music streaming service
- 7,558,834 observations of 15 variables (genre_id, ts_listen, media_id, album_id, 
context_type, release_date, platform_name, platform_family, media_duration, 
listen_type, user_gender, user_id, artist_id, user_age, is_listened).
<br>

<b>Models</b>:<br>
- Three different models: content-based (item-based), collaborative filtering (user-based) and matrix factorization are used<br>
- The purpose is to predict whether users listened to the track proposed to them or not
<br>

<b>Results</b>:<br><br>
The results shown in this section refer to the test dataset, which was build based on the full data set. The test data set consists of 20% of all observations. 
The table below shows all the results of the accurracy metrics summarized for each of our models.<br>
<img src="https://github.com/leiagnes/rec/blob/main/models_performance.jpg" height="100"><br><br>
We can state that the SVD model has the lowest RMSE with 0.4123. Moreover, its AUC value is the highest, 
which makes it the most appropriate model in terms of good prediction respectively accuracy. 
However, as far as the MAE is concerned, our content-based model performs best with a result of 0.3121, 
closely followed by our user-based collaborative filtering model. 
However, we would classify the latter model as the worst model in terms of accuracy. Its RMSE is the highest, while its AUC is the lowest.<br>
<img src="https://github.com/leiagnes/rec/blob/main/models_ROC.jpg" height="400"><br><br>
As mentioned above, the SVD model has the highest AUC and its ROC curve is also the smoothest.
