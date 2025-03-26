import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.neighbors import NearestNeighbors

class MovieRecommendationSystem:
    def __init__(self, dataset_path, model_path):
        # Load dataset and XGBoost model
        self.df = pd.read_csv(dataset_path)
        self.model = xgb.XGBRegressor()  # Use XGBoost regressor
        self.model.load_model(model_path)  # Load the trained XGBoost model
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

    def predict_similarity(self, selected_movie_name, filtered_df, model_type):
        # Normalize the selected movie name to lowercase for comparison
        selected_movie_name_lower = selected_movie_name.lower()
        
        # Find the movie in the dataframe based on lowercase title comparison
        selected_movie = self.df[self.df['Title_Lower'] == selected_movie_name_lower]
        
        if selected_movie.empty:
            print("Movie not found in the dataset. Please try again with a different title.")
            return pd.DataFrame()  # Return an empty DataFrame if the movie is not found

        if model_type == 'xgboost':
            selected_movie_features = selected_movie[self.feature_columns].values
            selected_movie_features_scaled = self.scaler.transform(selected_movie_features)
        
            similarity_scores = []

            for i in range(len(filtered_df)):
                movie_features = self.X_scaled[i]
                pair_features = np.concatenate([selected_movie_features_scaled.flatten(), movie_features.flatten()])
                
                # Predict similarity for the movie pair
                predicted_similarity = self.model.predict(pair_features.reshape(1, -1))
                similarity_scores.append(predicted_similarity[0])

            # Rank movies based on similarity score
            similarity_df = pd.DataFrame({
                'Title': filtered_df['Title'],
                'Similarity Score': similarity_scores
            })

            # Return top 5 most similar movies
            top_5_similar_movies = similarity_df.sort_values(by='Similarity Score', ascending=False).head(6)[1:]

        elif model_type == 'knn':
            scaler = StandardScaler()
            movie_features = filtered_df[self.feature_columns].values
            movie_features_scaled = scaler.fit_transform(movie_features)

            # Step 4: Implement KNN for finding similar movies
            knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 neighbors to exclude the selected movie itself
            knn.fit(movie_features_scaled)

            selected_movie_idx = self.df[self.df['Title_Lower'] == selected_movie_name_lower].index[0]
            selected_movie_features = movie_features_scaled[selected_movie_idx].reshape(1, -1)

            # Find the top 5 most similar movies
            distances, indices = knn.kneighbors(selected_movie_features, n_neighbors=6)

            # Convert distances to similarity scores (higher similarity means smaller distance)
            similarity_scores = 1 - distances / np.max(distances)

            # Step 6: Rank movies based on similarity score
            top_5_indices = indices[0][1:]  # Skip the first one (the movie itself)
            top_5_scores = similarity_scores[0][1:]
            top_5_movies = filtered_df.iloc[top_5_indices][['Title']].copy()
            
            top_5_similar_movies = pd.DataFrame({
                'Title': top_5_movies['Title'],
                'Similarity Score': top_5_scores
            })
        
        elif model_type == 'xgboost+knn':
                    # XGBoost predictions
            selected_movie_features = selected_movie[self.feature_columns].values
            selected_movie_features_scaled = self.scaler.transform(selected_movie_features)
            
            xgboost_similarity_scores = []
            for i in range(len(filtered_df)):
                movie_features = self.X_scaled[i]
                pair_features = np.concatenate([selected_movie_features_scaled.flatten(), movie_features.flatten()])
                
                # Predict similarity for the movie pair using XGBoost
                predicted_similarity = self.model.predict(pair_features.reshape(1, -1))
                xgboost_similarity_scores.append(predicted_similarity[0])
            
            # KNN predictions
            scaler_knn = StandardScaler()
            movie_features_knn = filtered_df[self.feature_columns].values
            movie_features_scaled_knn = scaler_knn.fit_transform(movie_features_knn)

            knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 neighbors to exclude the selected movie itself
            knn.fit(movie_features_scaled_knn)

            selected_movie_idx = filtered_df[filtered_df['Title_Lower'] == selected_movie_name].index[0]
            selected_movie_features_knn = movie_features_scaled_knn[selected_movie_idx].reshape(1, -1)

            # Find the top 5 most similar movies using KNN
            distances, indices = knn.kneighbors(selected_movie_features_knn, n_neighbors=6)

            # Convert distances to similarity scores (higher similarity means smaller distance)
            knn_similarity_scores = 1 - distances / np.max(distances)

            # Rank movies based on similarity score, skipping the first one (the selected movie itself)
            top_5_indices_knn = indices[0][1:]  # Skip the first one (the movie itself)
            top_5_scores_knn = knn_similarity_scores[0][1:]
            top_5_movies_knn = filtered_df.iloc[top_5_indices_knn][['Title']].copy()
            top_5_movies_knn['Similarity Score'] = top_5_scores_knn

            # Combine the similarity scores from both models (average or other method)
            similarity_df_xgboost = pd.DataFrame({
                'Title': filtered_df['Title'],
                'XGBoost Similarity Score': xgboost_similarity_scores
            })

            similarity_df_knn = pd.DataFrame({
                'Title': top_5_movies_knn['Title'],
                'KNN Similarity Score': top_5_movies_knn['Similarity Score']
            })

            # Merge both similarity dataframes
            combined_similarity = pd.merge(similarity_df_xgboost, similarity_df_knn, on='Title')

            # Combine the scores (e.g., average them)
            combined_similarity['Combined Similarity Score'] = (combined_similarity['XGBoost Similarity Score'] + combined_similarity['KNN Similarity Score']) / 2

            # Sort by the combined similarity score and return the top 5
            top_5_combined = combined_similarity.sort_values(by='Combined Similarity Score', ascending=False).head(5)

            top_5_similar_movies = top_5_combined[['Title', 'Combined Similarity Score']]


        return top_5_similar_movies

    def explain_similarity(self, selected_movie, similar_movie, feature_columns):
        explanation = []
        
        # Loop through the list of features to compare
        for feature in ['Genres', 'IMDb Rating', 'Oscars', 'Runtime (mins)']:
            if feature in selected_movie.columns and feature in similar_movie.columns:
                # Access the values of the selected movie and the similar movie
                val1 = selected_movie[feature].values[0]
                val2 = similar_movie[feature].values[0]
                
                # Handle the 'Genres' feature
                if feature == 'Genres':
                    # Handle cases where Genres is either a string or a list
                    try:
                        if isinstance(val1, str):
                            genres1 = set(ast.literal_eval(val1)) if val1.startswith("[") else set(val1.split(', '))
                        else:  # Assume it's a list if not a string
                            genres1 = set(val1)
                        
                        if isinstance(val2, str):
                            genres2 = set(ast.literal_eval(val2)) if val2.startswith("[") else set(val2.split(', '))
                        else:  # Assume it's a list if not a string
                            genres2 = set(val2)
                        
                        # Find common genres between the two movies
                        common_genres = genres1 & genres2
                        if common_genres:
                            explanation.append(f"Similar genre(s): {common_genres}")
                    except:
                        explanation.append("Error processing genres.")
                
                # Handle 'IMDb Rating' and 'Oscars' with a close value condition
                elif feature in ['IMDb Rating', 'Oscars'] and abs(val1 - val2) < 1.0:
                    explanation.append(f"Close {feature}: {val1} vs {val2}")
                
                # Handle 'Runtime (mins)' with a close value condition
                elif feature == 'Runtime (mins)' and abs(val1 - val2) < 20:
                    explanation.append(f"Close runtime: {val1} mins vs {val2} mins")
                    
        return "; ".join(explanation)


    def recommend_movies(self, model):
        # Get user input
        selected_movie_name, genre_filter, rating_range, runtime_range = self.get_user_input()

        # Apply filters based on user input
        filtered_df = self.apply_filters(selected_movie_name, genre_filter, rating_range, runtime_range)

        # Predict similarity and get top 5 recommendations
        top_5_similar_movies = self.predict_similarity(selected_movie_name, filtered_df, model)

        # Display the results
        if not top_5_similar_movies.empty:
            print("\nTop 5 recommended movies based on your selection and filters:")
            print(top_5_similar_movies)
            
            # Generate explanations for the top 5 similar movies
            explanations = []
            
            # Get the selected movie to pass it to the explain_similarity function
            selected_movie = self.df[self.df['Title_Lower'] == selected_movie_name.lower()]

            for index, row in top_5_similar_movies.iterrows():
                movie_title = row['Title']
                similar_movie = filtered_df[filtered_df['Title'] == movie_title]
                
                # Get the explanation for each similar movie
                explanation = self.explain_similarity(selected_movie, similar_movie, self.feature_columns)
                explanations.append((movie_title, explanation))
            
            # Display the explanations
            for movie_title, explanation in explanations:
                print(f"Movie: {movie_title}")
                print(f"Explanation: {explanation}")
                print("="*50)
        else:
            print("No movies were found matching your criteria.")


if __name__ == "__main__":
    # Create an instance of the MovieRecommendationSystem
    movie_recommender = MovieRecommendationSystem(dataset_path='top100_preprocessed.csv', model_path='xgboost_movie_similarity_model.json')
    
    # Get movie recommendations
    # model = 'xgboost'
    model = 'knn'
    # model = 'xgboost+knn'
    movie_recommender.recommend_movies(model)
