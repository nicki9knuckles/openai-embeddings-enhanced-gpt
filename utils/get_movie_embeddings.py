import pandas as pd

from utils.cache import load_embedding_cache
from utils.embeddings import embedding_from_string


def get_movie_embeddings():
    embedding_cache_path = "movie_embeddings.pkl"
    embedding_cache = load_embedding_cache(embedding_cache_path)
    movie_data = pd.read_pickle("final_movie_data.pkl")
    movie_embeddings = {}

    for index, row in movie_data.iterrows():
        plot = row["movie_info"]
        movie_title = row["movie_title"]
        tomatometer_rating = row["tomatometer_rating"]
        original_release_date = row["original_release_date"]
        if (
            plot == ""
            or plot is None
            or (isinstance(plot, float) and math.isnan(plot))
            or pd.isna(movie_title)
            or pd.isna(tomatometer_rating)
            or pd.isna(original_release_date)
        ):
            # print("Skipping movie due to missing properties.")
            continue

        combined_string = f"title: {movie_title}, rating: {tomatometer_rating}, release: {original_release_date}, plot: {plot}"

        movie_embeddings[combined_string] = embedding_from_string(
            embedding_cache_path,
            combined_string,
            embedding_cache,
            model="text-embedding-ada-002",
        )

    return movie_embeddings
