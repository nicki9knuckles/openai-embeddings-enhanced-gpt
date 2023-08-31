import pandas as pd
import pickle


def load_embedding_cache(embedding_cache_path="movie_embeddings.pkl"):
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache
