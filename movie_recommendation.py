import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class MovieRecommendationSystem:
    def __init__(self, dataset_path, model_path):
        # Load dataset and model
        self.df = pd.read_csv(dataset_path)
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()
        
        # Feature columns for the model
        self.feature_columns = [
            'IMDb Rating', 'Runtime (mins)', 'Year', 'Actors_Stars', 'Direction_Stars', 'Screenplay_Stars',
            'Oscars', 'Oscar_Nominations', 'BAFTA_Awards', 'BAFTA_Nominations', 'Golden_Globes', 'Golden_Globe_Nominations',
            'Action', 'Adventure', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
            'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'
        ]
        
        # Preprocess dataset
        self.df['Title_Lower'] = self.df['Title'].str.lower()  # Add a lowercase version of the title for comparison
        self.X = self.df[self.feature_columns].values
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def get_user_input(self):
        print("Welcome to the Movie Recommendation System!")
        selected_movie_name = input("Enter the movie title you want to base your recommendations on: ").strip().lower()  # Convert to lowercase
        
        # Genre filter (optional, case insensitive)
        genre_filter_input = input("Enter genres you're interested in (comma separated, e.g., Drama, Comedy), or press Enter to skip: ")
        genre_filter = [genre.strip().lower() for genre in genre_filter_input.split(',')] if genre_filter_input else []

        # IMDb Rating range filter (optional, accepting 'number-number' or just Enter to skip)
        rating_input = input("Enter the IMDb rating range (e.g., 7.0-9.0), or press Enter to skip: ")
        rating_range = self.parse_range_input(rating_input)

        # Runtime range filter (optional, accepting 'number-number' or just Enter to skip)
        runtime_input = input("Enter the runtime range (in minutes, e.g., 90-150), or press Enter to skip: ")
        runtime_range = self.parse_range_input(runtime_input, is_runtime=True)
        
        return selected_movie_name, genre_filter, rating_range, runtime_range

    def parse_range_input(self, input_str, is_runtime=False):
        if input_str == '':
            return (None, None)
        try:
            start, end = map(float, input_str.split('-'))
            if is_runtime:
                # Ensure the runtime is a valid number (positive integers)
                if start < 0 or end < 0:
                    print("Invalid runtime range. Using default.")
                    return (None, None)
            return (start, end)
        except ValueError:
            print(f"Invalid input format for {'runtime' if is_runtime else 'rating'} range. Skipping filter.")
            return (None, None)

    def apply_filters(self, selected_movie_name, genre_filter, rating_range, runtime_range):
        filtered_df = self.df
        
        # Apply Genre Filter if provided (case insensitive)
        if genre_filter:
            filtered_df = filtered_df[filtered_df['Genres'].apply(lambda x: any(genre.lower() in x.lower() for genre in genre_filter))]

        
        # Apply IMDb Rating Filter if provided
        if rating_range[0] is not None and rating_range[1] is not None:
            filtered_df = filtered_df[(filtered_df['IMDb Rating'] >= rating_range[0]) & (filtered_df['IMDb Rating'] <= rating_range[1])]
        
        # Apply Runtime Filter if provided
        if runtime_range[0] is not None and runtime_range[1] is not None:
            filtered_df = filtered_df[(filtered_df['Runtime (mins)'] >= runtime_range[0]) & (filtered_df['Runtime (mins)'] <= runtime_range[1])]

        return filtered_df

    def predict_similarity(self, selected_movie_name, filtered_df):
        # Normalize the selected movie name to lowercase for comparison
        selected_movie_name_lower = selected_movie_name.lower()
        
        # Find the movie in the dataframe based on lowercase title comparison
        selected_movie = self.df[self.df['Title_Lower'] == selected_movie_name_lower]
        
        if selected_movie.empty:
            print("Movie not found in the dataset. Please try again with a different title.")
            return pd.DataFrame()  # Return an empty DataFrame if the movie is not found
        
        selected_movie_features = selected_movie[self.feature_columns].values
        selected_movie_features_scaled = self.scaler.transform(selected_movie_features)
        
        similarity_scores = []

        for i in range(len(filtered_df)):
            movie_features = self.X_scaled[i]
            pair_features = np.concatenate([selected_movie_features_scaled.flatten(), movie_features.flatten()])
            
            # Predict similarity for the movie pair
            predicted_similarity = self.model.predict(pair_features.reshape(1, -1))
            similarity_scores.append(predicted_similarity[0][0])

        # Step 4: Rank movies based on similarity score
        similarity_df = pd.DataFrame({
            'Title': filtered_df['Title'],
            'Similarity Score': similarity_scores
        })

        # Return top 5 most similar movies
        top_5_similar_movies = similarity_df.sort_values(by='Similarity Score', ascending=False).head(6)[1:]
        return top_5_similar_movies

    def recommend_movies(self):
        # Get user input
        selected_movie_name, genre_filter, rating_range, runtime_range = self.get_user_input()

        # Apply filters based on user input
        filtered_df = self.apply_filters(selected_movie_name, genre_filter, rating_range, runtime_range)

        # Predict similarity and get top 5 recommendations
        top_5_similar_movies = self.predict_similarity(selected_movie_name, filtered_df)

        # Display the results
        if not top_5_similar_movies.empty:
            print("\nTop 5 recommended movies based on your selection and filters:")
            print(top_5_similar_movies)
        else:
            print("No movies were found matching your criteria.")

if __name__ == "__main__":
    # Create an instance of the MovieRecommendationSystem
    movie_recommender = MovieRecommendationSystem(dataset_path='top100_preprocessed.csv', model_path='movie_similarity_model.h5')
    
    # Get movie recommendations
    movie_recommender.recommend_movies()
