import openai
import tiktoken
import os
from dotenv import load_dotenv
import pandas as pd

from utils.cache import load_embedding_cache
from utils.embeddings import (
    embedding_from_string,
    ask_embedding_store,
)


MAX_CONTEXT_WINDOW = 4097
MINIMUM_RESPONSE_SPACE = 1000
MAX_PROMPT_SIZE = MAX_CONTEXT_WINDOW - MINIMUM_RESPONSE_SPACE
embedding_cache_path = "movie_embeddings.pkl"
chat_model = "gpt-3.5-turbo"
embedding_enc = tiktoken.encoding_for_model("text-embedding-ada-002")
enc = tiktoken.encoding_for_model(chat_model)


def main():
    embedding_cache = load_embedding_cache(embedding_cache_path)
    movie_data = pd.read_pickle("final_movie_data.pkl")

    plot_embedding = {}

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

        # print(f"combined_string: {combined_string}")
        plot_embedding[combined_string] = embedding_from_string(
            embedding_cache_path,
            combined_string,
            embedding_cache,
            model="text-embedding-ada-002",
        )

    # first_key = next(iter(embedding_cache))
    # first_value = embedding_cache[first_key]

    # print(f"First key: {first_key}")
    # print(f"First value: {first_value}")

    ask_embedding_store(
        MAX_PROMPT_SIZE,
        chat_model,
        enc,
        "Can you suggest 2 or 3 highly rated horror movies from 2023, and tell me a bit about their plots?",
        plot_embedding,
        5,
    )


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
