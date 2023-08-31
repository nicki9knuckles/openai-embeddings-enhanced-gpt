import openai
import tiktoken
import os
from dotenv import load_dotenv
import pandas as pd

# import logging
# import tempfile


from utils.cache import load_embedding_cache
from utils.data import simplify_data
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
    movie_data = simplify_data("rotten_tomatoes_movies.csv")
    # Print the first 5 rows to check
    # print(df.head().T)
    # unique_values = df["audience_status"].unique()
    # print(f"Unique values in 'audience_status': {unique_values}")

    num_rows = movie_data.shape[0]
    # print(f"The CSV file has {num_rows} rows.")
    # print(f"embedding_cache {embedding_cache}")

    movie_plots = (
        # drop any rows where "movie_info" is NaN before converting the remaining values to strings.
        movie_data.dropna(subset=["movie_info"])["movie_info"]
        .astype(str)
        .values
    )

    plot_embedding = {}
    for plot in movie_plots:
        plot_embedding[plot] = embedding_from_string(
            embedding_cache_path, plot, embedding_cache, model="text-embedding-ada-002"
        )

    # first_key, first_value = list(plot_embedding.items())[0]
    # print("First plot:", first_key)
    # print("First embedding:", first_value)
    ask_embedding_store(
        MAX_PROMPT_SIZE,
        chat_model,
        enc,
        "Do you know about any movies that have to do with heartbreak? If so can you tell me the plot?",
        plot_embedding,
        15,
    )


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
