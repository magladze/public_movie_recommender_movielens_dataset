import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Data Collection and Preprocessing
# Load the MovieLens dataset (you can download it from https://grouplens.org/datasets/movielens/)
# For simplicity, we will use the smaller dataset (ml-latest-small)

# Load the data into pandas DataFrame
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')

# Display the first few rows of the datasets
print(ratings_df.head())
print(movies_df.head())

# Step 2: Feature Engineering
# We will use collaborative filtering, so we need to create user-item interaction matrices
# For this, we will use the Surprise library, which requires data in a specific format

# Define a reader with the rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load the data into the Surprise dataset
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Step 3: Model Selection
# We will use the SVD (Singular Value Decomposition) algorithm for collaborative filtering
algo = SVD()

# Step 4: Training the Model
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the model on the training data
algo.fit(trainset)

# Step 5: Evaluation and Hyperparameter Tuning
# Evaluate the model on the test data
predictions = algo.test(testset)

# Calculate and print RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')


# Step 6: Deployment and Testing
# We can now use the trained model to make movie recommendations for a specific user
def get_movie_recommendations(user_id, num_recommendations=10):
    # Get a list of all movie IDs
    all_movie_ids = movies_df['movieId'].unique()

    # Predict ratings for all movies for the given user
    user_ratings = [(movie_id, algo.predict(user_id, movie_id).est) for movie_id in all_movie_ids]

    # Sort the predicted ratings in descending order and get the top recommendations
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = user_ratings[:num_recommendations]

    # Get the movie titles for the top recommendations
    top_movie_ids = [movie_id for movie_id, rating in top_recommendations]
    top_movies = movies_df[movies_df['movieId'].isin(top_movie_ids)]

    return top_movies



# ********************************************************************************************

if __name__ == '__main__':
    # Test the recommendation system for a specific user
    user_id = 1  # Example user ID
    recommendations = get_movie_recommendations(user_id)
    print(f"Top movie recommendations for user {user_id}:")
    print(recommendations[['title']])