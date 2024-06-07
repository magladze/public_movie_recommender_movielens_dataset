 # Movie Recommender Tool

## Overview

This project is a movie recommendation system built using the MovieLens dataset. It employs collaborative filtering with matrix factorization, implemented using the `Surprise` library, which is a Python scikit for building and analyzing recommender systems. The system is designed to predict user preferences for movies and provide personalized recommendations.

## Table of Contents

1. [Introduction](#introduction)
2. [History of the MovieLens Dataset](#history-of-the-movielens-dataset)
3. [Libraries Used](#libraries-used)
4. [Data Science Models](#data-science-models)
5. [Code Explanation](#code-explanation)
6. [Use Cases](#use-cases)
7. [Deployment to Google Cloud](#deployment-to-google-cloud)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project demonstrates how to build a movie recommendation system using collaborative filtering. The system uses the MovieLens dataset and the `Surprise` library to predict user ratings for movies and provide recommendations.

## History of the MovieLens Dataset

The MovieLens dataset is a collection of movie ratings collected by the GroupLens Research Project at the University of Minnesota. It is widely used in the research community for benchmarking recommendation algorithms. The dataset includes millions of ratings and tag applications across thousands of movies. The MovieLens datasets are available in various sizes, with the "ml-latest-small" dataset being used in this project for simplicity.

## Libraries Used

### Pandas
- **History**: Developed by Wes McKinney in 2008, Pandas is a powerful and flexible data manipulation library in Python.
- **Explanation**: Used for data manipulation and analysis, providing data structures like DataFrames.

### NumPy
- **History**: Created in 2005 by Travis Oliphant, NumPy is a fundamental package for scientific computing in Python.
- **Explanation**: Used for numerical operations and handling arrays.

### Surprise
- **History**: Developed by Nicolas Hug, the Surprise library is a Python scikit for building and analyzing recommender systems.
- **Explanation**: Provides tools to evaluate, compare, and build recommendation algorithms.

## Data Science Models

### Collaborative Filtering
- **Explanation**: A method used to make automatic predictions about a user's interests by collecting preferences from many users. The assumption is that users who agreed in the past will agree in the future.

### Singular Value Decomposition (SVD)
- **Explanation**: A matrix factorization technique used in collaborative filtering to reduce the dimensionality of the user-item interaction matrix, making it easier to predict missing values (ratings).

## Code Explanation

```python
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Data Collection and Preprocessing
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')

# Feature Engineering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Model Selection
algo = SVD()

# Training the Model
trainset, testset = train_test_split(data, test_size=0.2)
algo.fit(trainset)

# Evaluation and Hyperparameter Tuning
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Deployment and Testing
def get_movie_recommendations(user_id, num_recommendations=10):
    all_movie_ids = movies_df['movieId'].unique()
    user_ratings = [(movie_id, algo.predict(user_id, movie_id).est) for movie_id in all_movie_ids]
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = user_ratings[:num_recommendations]
    top_movie_ids = [movie_id for movie_id, rating in top_recommendations]
    top_movies = movies_df[movies_df['movieId'].isin(top_movie_ids)]
    return top_movies

user_id = 1  # Example user ID
recommendations = get_movie_recommendations(user_id)
print(f"Top movie recommendations for user {user_id}:")
print(recommendations[['title']])
```

### Steps Explained:
1. **Data Collection and Preprocessing**: Load the MovieLens dataset into pandas DataFrames.
2. **Feature Engineering**: Prepare the data for the `Surprise` library by creating a `Reader` and loading the data into a `Dataset`.
3. **Model Selection**: Choose the SVD algorithm for collaborative filtering.
4. **Training the Model**: Split the data into training and testing sets, then train the model.
5. **Evaluation and Hyperparameter Tuning**: Evaluate the model using RMSE on the test data.
6. **Deployment and Testing**: Define a function to get movie recommendations for a specific user by predicting ratings for all movies and selecting the top recommendations.

## Use Cases

- **Personalized Movie Recommendations**: Provide users with movie suggestions based on their past ratings and preferences.
- **Content Filtering**: Enhance user experience on streaming platforms by recommending relevant content.
- **Market Basket Analysis**: Identify patterns in user behavior and preferences for targeted marketing.

## Deployment to Google Cloud

### Steps to Deploy on Google Cloud:

1. **Google Cloud Platform (GCP) Setup**: Create a GCP account and set up a project to host the recommendation system.
2. **Cloud Storage**: Store the model artifacts, movie dataset, and any other necessary files on Google Cloud Storage for easy access.
3. **Compute Engine**: Deploy the Python Flask/FastAPI application on a Google Compute Engine instance to serve the recommendation API.
4. **Cloud Endpoints**: Use Google Cloud Endpoints to manage and secure the API, enabling easy integration with other services.
5. **Load Balancer**: Set up a load balancer to distribute incoming traffic and ensure high availability and scalability of the recommendation system.
6. **Monitoring & Logging**: Implement Stackdriver for monitoring system performance, logging errors, and tracking usage metrics.

## Installation

To run this project locally, you need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install scikit-surprise pandas numpy
```

## Usage

1. Download the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/) and extract it.
2. Place the `ratings.csv` and `movies.csv` files in the `data` directory.
3. Run the Python script to see the movie recommendations for a specific user.

```bash
python movie_recommender_system.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


---

Feel free to explore the code and enhance the recommendation system with additional features or improved algorithms. Happy coding! 