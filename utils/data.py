import pandas as pd
import math


def simplify_data(data_path="final_movie_data.pkl"):
    df = pd.read_pickle(data_path)
    # print(df.head().T)

    selected_columns = [
        "movie_info",
        "original_release_date",
        "movie_title",
        "tomatometer_rating",
        "themoviedb_link",
        "rotten_tomatoes_link",
    ]
    selected_columns_df = df[selected_columns]

    # Sort the DataFrame by "original_release_date"
    sorted_df = selected_columns_df.sort_values(
        by="original_release_date", ascending=False
    )

    # Filter rows with an original_release_date of 1970 and newer
    filtered_df = sorted_df[sorted_df["original_release_date"] >= "2018-01-01"]

    # Filtering the DataFrame
    filtered_movie_data = filtered_df.dropna(subset=["movie_info"])

    # Converting 'movie_info' to string type, if needed
    filtered_movie_data.loc[:, "movie_info"] = filtered_movie_data["movie_info"].astype(
        str
    )
    dropped_nan_items = filtered_movie_data.dropna(subset=["movie_info"])

    dropped_nan_items.to_pickle(data_path)

    return dropped_nan_items


simplify_data()
