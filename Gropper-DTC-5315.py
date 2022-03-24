
# coding: utf-8

# # A Movie Recommendation Service
# ### Source: https://www.codementor.io/spark/tutorial/building-a-recommender-with-apache-spark-python-example-app-part1

# #### Create a SparkContext configured for local mode

# In[1]:


import pyspark
sc = pyspark.SparkContext('local[*]')


# #### File download
# Small: 100,000 ratings and 2,488 tag applications applied to 8,570 movies by 706 users. Last updated 4/2015.   
# Full: 21,000,000 ratings and 470,000 tag applications applied to 27,000 movies by 230,000 users. Last updated 4/2015.

# In[2]:


complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'


# #### Download location(s)

# In[3]:


import os
datasets_path = os.path.join('/home/jovyan', 'work')
complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')


# #### Getting file(s)

# In[4]:


import urllib.request

small_f = urllib.request.urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urllib.request.urlretrieve (complete_dataset_url, complete_dataset_path)


# #### Extracting file(s)

# In[5]:


import zipfile

with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)
    
with zipfile.ZipFile(complete_dataset_path, "r") as z:
    z.extractall(datasets_path)
print('Done')


# ## Loading and parsing datasets
# Now we are ready to read in each of the files and create an RDD consisting of parsed lines. 
# 
# Each line in the ratings dataset (ratings.csv) is formatted as: 
# + userId,movieId,rating,timestamp 
# 
# Each line in the movies (movies.csv) dataset is formatted as:
# + movieId,title,genres 
# 
# The format of these files is uniform and simple, so we can use Python split() to parse their lines once they are loaded into RDDs. Parsing the movies and ratings files yields two RDDs: 
# + For each line in the ratings dataset, we create a tuple of (UserID, MovieID, Rating). We drop the timestamp because we do not need it for this recommender.
# + For each line in the movies dataset, we create a tuple of (MovieID, Title). We drop the genres because we do not use them for this recommender.

# #### ratings.csv

# In[6]:


# left off here, I believe that I follow the tutorial exactly until I reach "explanation of execution" on here, which is from A8,
# there I should continue on with the tutorial section "Using the complete dataset to build the final model" until "persisting the model"
# removing complete_dataset stuff and continueing: 2/9/22
small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
# Parse
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

print ('There are {} recommendations in the complete dataset'.format(small_ratings_data.count()))
small_ratings_data.take(3)


# #### movies.csv

# In[7]:


small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    

small_movies_titles = small_movies_data.map(lambda x: (int(x[0]),x[1]))
print ('There are {} movies in the small dataset'.format(small_movies_data.count()))
small_movies_data.take(3)
print('Done')


# ## Collaborative Filtering
# In Collaborative filtering we make predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption is that if a user A has the same opinion as a user B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a user chosen randomly. 
# 
# At first, people rate different items (like videos, images, games). Then, the system makes predictions about a user's rating for an item not rated yet. The new predictions are built upon the existing ratings of other users with similar ratings with the active user. In the image, the system predicts that the user will not like the video.
# 
# Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has the following parameters:
# 
# + numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
# + rank is the number of latent factors in the model.
# + iterations is the number of iterations to run.
# + lambda specifies the regularization parameter in ALS.
# + implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
# + alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

# #### Selecting ALS parameters using the small dataset
# In order to determine the best ALS parameters, we will use the small dataset. We need first to split it into train, validation, and test datasets.

# In[8]:


# source uses seed=0L, which is the previous version of python (2.x)
# 0L should be written as 0 from now on
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


# #### Training phase

# In[9]:


from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print ('The best model was trained with rank %s' % best_rank)
print(predictions.take(3))


# Basically we have the UserID, the MovieID, and the Rating, as we have in our ratings dataset. In this case the predictions third element, the rating for that movie and user, is the predicted by our ALS model.
# 
# Then we join these with our validation data (the one that includes ratings) and the result looks as follows:

# In[10]:


rates_and_preds.take(3)


# To that, we apply a squared difference and the we use the mean() action to get the MSE and apply sqrt.
# 
# Finally we test the selected model.

# In[11]:


model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print('For testing data the RMSE is %s' % (error))


# ## Using the complete dataset to build the final model
# Due to the limitations of virtual machine, we keep using the small dataset instead of complete dataset
# 
# We need first to split it into training and test datasets.

# In[12]:


# Load the complete dataset file
complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
#print('yeet')
# Parse
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print("There are %s recommendations in the complete dataset" % (complete_ratings_data.count()))
# yay working!


# Now we are ready to train the recommender model.

# In[13]:


training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)

complete_model = ALS.train(training_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=regularization_parameter)


# Now we test this on our testing set.

# In[15]:


test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
print('Done up to error')
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print('error done')

print ('For testing data the RMSE is {}'.format(error))
#this and all above are running great, takes a while


# We can see how we got a more accurate recommender when using a much larger dataset.

# # Explanation of Execution:

# ## How to make recommendations
# Although we aim at building an online movie recommender, now that we know how to have our recommender model ready, we can give it a try providing some movie recommendations. This will help us coding the recommending engine later on when building the web service, and will explain how to use the model in any other circumstances.
# 
# When using collaborative filtering, getting recommendations is not as simple as predicting for the new entries using a previously generated model. Instead, we need to train again the model but including the new user preferences in order to compare them with other users in the dataset. That is, the recommender needs to be trained every time we have new user ratings (although a single model can be used by multiple users of course!). This makes the process expensive, and it is one of the reasons why scalability is a problem (and Spark a solution!). Once we have our model trained, we can reuse it to obtain top recomendations for a given user or an individual rating for a particular movie. These are less costly operations than training the model itself.

# In[16]:


complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))
    
print("There are %s movies in the complete dataset" % (complete_movies_titles.count()))


# Another thing we want to do, is give recommendations of movies with a certain minimum number of ratings. For that, we need to count the number of ratings per movie.

# In[17]:


def get_counts_and_averages(ID_and_ratings_tuple):
   nratings = len(ID_and_ratings_tuple[1])
   return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# ### Adding new user ratings
# Now we need to rate some movies for the new user. We will put them in a new RDD and we will use the user ID 0, that is not assigned in the MovieLens dataset. Check the dataset movies file for ID to Tittle assignment (so you know what movies are you actually rating).

# In[27]:


new_user_ID = 0

# The format of each line is (userID, movieID, rating)

# ###################################################
# Keep the userID, but Replace movieID, rating, title
# ###################################################

# # Find 10 movies you have watched in the past
# # Put your OWN ratings
#ORIGINAL
# new_user_ratings = [
#      (0,260,4), # Star Wars (1977)
#      (0,1,3), # Toy Story (1995)
#      (0,16,3), # Casino (1995)
#      (0,25,4), # Leaving Las Vegas (1995)
#      (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
#      (0,335,1), # Flintstones, The (1994)
#      (0,379,1), # Timecop (1994)
#      (0,296,3), # Pulp Fiction (1994)
#      (0,858,5), # Godfather, The (1972)
#      (0,50,4) # Usual Suspects, The (1995)
#     ]

# #MODIFIED MY OWN DTC
# new_user_ratings = [
#      (0,260,4.5), # Star Wars (1977)
#      (0,858,5), # Godfather, The (1972)
#      (0,1374,3), #Star Trek II: The Wrath of Khan (1982)
#      (0,1387,4.5), # Jaws (1975)
#      (0,1544,5), # Lost World: Jurassic Park, The (1997)
#      (0,1562,3), # Batman & Robin (1997)
#      (0,1704,5), # Good Will Hunting (1997)
#      (0,1707,2.5), # Home Alone 3 (1997)
#      (0,2080,4.5), # Lady and the Tramp (1955) 
#      (0,88140,5) # Captain America: The First Avenger (2011)
#     ]

# Now a family members ratings:
new_user_ratings = [
    (0,1028,4), #Mary Poppins (1964)
    (0,2090,4.5), #Rescuers, The (1977)
    (0,3786,4), #But I'm a Cheerleader (1999)
    (0,4700,5), #Princess Diaries, The (2001)
    (0,135536,4), #Suicide Squad (2016)   
    (0,5990,4), #Pinocchio (2002)
    (0,8808,5), #Princess Diaries 2: Royal Engagement, The (2004)
    (0,168366,4), #Beauty and the Beast (2017)
    (0,73017,3.5), #Sherlock Holmes (2009)
    (0,155743,4), #My Big Fat Greek Wedding 2 (2016)
]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print ('New user ratings: {}'.format(new_user_ratings_RDD.take(10)))


# Now we add them to the data we will use to train our recommender model. We use Spark's union() transformation for this.

# In[28]:


#sample_70 = complete_ratings_data.sample(False, 0.70, 42)
#complete_data_with_new_ratings_RDD = sample_70.union(new_user_ratings_RDD)
#can toggle above on and off for pulling just 70% of the data

complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)


# And finally we train the ALS model using all the parameters we selected before (when using the small dataset).

# In[29]:


from time import time

t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed,
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0

print ('New model trained in {} seconds'.format(round(tt,3)))


# ## Getting top recommendations
# Let's now get some recommendations! For that we will get an RDD with all the movies the new user hasn't rated yet. We will combine them together with the model to predict ratings.

# In[30]:


new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)


# We have our recommendations ready. Now we can print out the 25 movies with the highest predicted ratings. And join them with the movies RDD to get the titles, and ratings count in order to get movies with a minimum number of counts. First we will do the join and see what does the result looks like.

# In[31]:


# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD =     new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)
#works, takes about 10 minutes


# So we need to flat this down a bit in order to have (Title, Rating, Ratings Count).

# In[32]:


new_user_recommendations_rating_title_and_count_RDD =     new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))


# Finally, get the highest rated recommendations for the new user, filtering out movies with less than 25 ratings

# In[33]:


# the "r[2] >= 25" is how many reviews each movie must have. the "takeordered(25" is how many recommended movies you want to see
#this is what needs to be changed for the DTC responses
#----Scenario 1 FULL dataset, filtering out movies with less than 25 ratings
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(15, key=lambda x: -x[1])
##################################################################### separating scenarios
#----Scenario 2 FULL dataset, filtering out movies with less than 100 ratings
top_movies2 = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=100).takeOrdered(15, key=lambda x: -x[1])


print ('TOP recommended movies (with more than 25 reviews):\n{}'.format('\n'.join(map(str, top_movies))))
print ('TOP recommended movies (with more than 100 reviews):\n{}'.format('\n'.join(map(str, top_movies2))))# just for ease of running


# ## Getting individual ratings
# Another useful usecase is getting the predicted rating for a particular movie for a given user. The process is similar to the previous retreival of top recommendations but, instead of using predcitAll with every single movie the user hasn't rated yet, we will just pass the method a single entry with the movie we want to predict the rating for.

# In[25]:


my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
individual_movie_rating_RDD.take(1)


# In[ ]:


#NOT SURE IF I NEED THE BELOW SECTION:


# ## Persisting the model
# Optionally, we might want to persist the base model for later use in our on-line recommendations. Although a new model is generated everytime we have new user ratings, it might be worth it to store the current one, in order to save time when starting up the server, etc. We might also save time if we persist some of the RDDs we have generated, specially those that took longer to process. For example, the following lines save and load a ALS model.

# In[26]:


from pyspark.mllib.recommendation import MatrixFactorizationModel

model_path = os.path.join('..', 'models', 'movie_lens_als')

# Save and load model
model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)


# Among other things, you will see in your filesystem that there are folder with product and user data into Parquet format files.
# 
# ## Genre and other fields
# We haven't used the genre and timestamp fields in order to simplify the transformations and the whole tutorial. Incorporating them doesn't represent any problem. A good use could be filtering recommendations by any of them (e.g. recommendations by genre, or recent recommendations) like we have done with the minimum number of ratings.

# In[ ]:


# all above is fully running!

