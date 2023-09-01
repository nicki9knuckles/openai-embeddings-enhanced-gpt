import pandas as pd
import pickle

# Load CSV file into DataFrame
csv_data = pd.read_csv("rotten_tomatoes_movies.csv")

# Load pickle file into DataFrame
with open("tmdb_movies_2023.pkl", "rb") as f:
    pickle_data_list = pickle.load(f)
pickle_data = pd.DataFrame(pickle_data_list)

pickle_data.rename(
    columns={
        "overview": "movie_info",
        "release_date": "original_release_date",
        "title": "movie_title",
        "vote_average": "tomatometer_rating",
        "id": "themoviedb_link",
    },
    inplace=True,
)


csv_data_filtered = csv_data[
    [
        "movie_info",
        "original_release_date",
        "movie_title",
        "tomatometer_rating",
        "rotten_tomatoes_link",
    ]
]
pickle_data_filtered = pickle_data[
    [
        "movie_info",
        "original_release_date",
        "movie_title",
        "tomatometer_rating",
        "themoviedb_link",
    ]
]

combined_data = pd.concat([csv_data_filtered, pickle_data_filtered], ignore_index=True)

combined_data.to_pickle("combined_data.pkl")

# Apply a function to scale down ratings to a max of 10 if they are greater than 10
combined_data["tomatometer_rating"] = combined_data["tomatometer_rating"].apply(
    lambda x: x / 10 if x > 10 else x
)

filtered_combined_data = combined_data[df["tomatometer_rating"] > 5]
# Save the filtered DataFrame, if needed
filtered_combined_data.to_pickle("filtered_combined_data.pkl")

print(filtered_combined_data.head().T)
