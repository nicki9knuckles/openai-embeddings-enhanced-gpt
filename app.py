import openai
from openai.embeddings_utils import cosine_similarity
import tiktoken
import os
from dotenv import load_dotenv
import pandas as pd
import pickle
from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging
import tempfile
from typing import Dict, List, Tuple, TypeVar
import itertools
import numpy as np


MAX_CONTEXT_WINDOW = 4097
MINIMUM_RESPONSE_SPACE = 1000
MAX_PROMPT_SIZE = MAX_CONTEXT_WINDOW - MINIMUM_RESPONSE_SPACE
T = TypeVar("T")  # Declare type variable
embedding_cache_path = "movie_embeddings.pkl"
chat_model = "gpt-3.5-turbo"
embedding_enc = tiktoken.encoding_for_model("text-embedding-ada-002")
enc = tiktoken.encoding_for_model(chat_model)


def load_embedding_cache(embedding_cache_path="movie_embeddings.pkl"):
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache


def simplify_data(csv_data_path="rotten_tomatoes_movies.csv"):
    df = pd.read_csv(csv_data_path)
    print(df.head().T)

    selected_columns = [
        "rotten_tomatoes_link",
        "movie_title",
        "movie_info",
        "critics_consensus",
        "genres",
        "original_release_date",
        "tomatometer_status",
        "audience_status",
    ]
    selected_columns_df = df[selected_columns]

    # Sort the DataFrame by "original_release_date"
    sorted_df = selected_columns_df.sort_values(
        by="original_release_date", ascending=False
    )

    # Filter rows with an original_release_date of 1970 and newer
    filtered_df = sorted_df[sorted_df["original_release_date"] >= "2018-01-01"]

    final_df = filtered_df[
        (filtered_df["tomatometer_status"].isin(["Fresh", "Certified-Fresh"]))
        & (filtered_df["audience_status"] == "Upright")
    ]
    return final_df


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages: List[Dict], model: str) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# Define a function to retrieve an embedding from the cache if present, otherwise request via OpenAI API
def embedding_from_string(string, embedding_cache, model="text-embedding-ada-002"):
    if (string, model) not in embedding_cache.keys():
        # Fetch the embedding
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")

        # Create a temporary file and write the updated cache to it
        temp_file_path = embedding_cache_path + ".tmp"
        with open(temp_file_path, "wb") as temp_embedding_cache_file:
            pickle.dump(embedding_cache, temp_embedding_cache_file)

        # If writing was successful, replace the old cache with the new one
        os.rename(temp_file_path, embedding_cache_path)
    # else:
    # print(f"USING CACHED EMBEDDING FOR {string[:20]}")

    try:
        return embedding_cache[(string, model)]
    except KeyError:
        print(f"KeyError for string--------------: {string[:5]}")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    if not isinstance(text, str):
        print(f"Warning: Invalid text {text} of type {type(text)}")
        return None  # or some other placeholder value

    # NOTE: remember to replace new line chars because of performance issues with OpenAI API
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]


def get_n_nearest_neighbors(
    query_embedding: List[float], embeddings: Dict[T, List[float]], n: int
) -> List[Tuple[T, float]]:
    """
    :param query_embedding: The embedding to find the nearest neighbors for
    :param embeddings: A dictionary of embeddings, where the keys are the entity type (e.g. Movie, Segment)
        and the values are the that entity's embeddings
    :param n: The number of nearest neighbors to return
    :return: A list of tuples, where the first element is the entity and the second element is the cosine
        similarity between -1 and 1
    """

    # This is not optimized for rapid indexing, but it's good enough for this example
    # If you're using this in production, you should use a more efficient vector datastore such as
    # those mentioned specifically by OpenAI here
    #
    #  https://platform.openai.com/docs/guides/embeddings/how-can-i-retrieve-k-nearest-embedding-vectors-quickly
    #
    #  * Pinecone, a fully managed vector database
    #  * Weaviate, an open-source vector search engine
    #  * Redis as a vector database
    #  * Qdrant, a vector search engine
    #  * Milvus, a vector database built for scalable similarity search
    #  * Chroma, an open-source embeddings store
    #

    target_embedding = np.array(query_embedding)

    similarities = [
        (segment, cosine_similarity(target_embedding, np.array(embedding)))
        for segment, embedding in embeddings.items()
    ]

    # Sort by similarity and get the top n results
    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    return nearest_neighbors


def ask_embedding_store(question: str, embeddings, max_documents: int) -> str:
    """
    Fetch necessary context from our embedding store, striving to fit the top max_documents
    into the context window (or fewer if the total token count exceeds the limit)

    :param question: The question to ask
    :param embeddings: A dictionary of Section objects to their corresponding embeddings
    :param max_documents: The maximum number of documents to use as context
    :return: GPT's response to the question given context provided in our embedding store
    """
    query_embedding = get_embedding(question)

    nearest_neighbors = get_n_nearest_neighbors(
        query_embedding, embeddings, max_documents
    )
    messages: Optional[List[Dict[str, str]]] = None

    base_token_count = num_tokens_from_messages(get_messages([], question), chat_model)

    token_counts = [
        len(enc.encode(document.replace("\n", " ")))
        for document, _ in nearest_neighbors
    ]
    cumulative_token_counts = list(itertools.accumulate(token_counts))
    indices_within_limit = [
        True
        for x in cumulative_token_counts
        if x <= (MAX_PROMPT_SIZE - base_token_count)
    ]
    most_messages_we_can_fit = len(indices_within_limit)

    context = [x[0] for x in nearest_neighbors[: most_messages_we_can_fit + 1]]

    messages = get_messages(context, question)

    #     print(f"Prompt: {messages[-1]['content']}")
    result = openai.ChatCompletion.create(model=chat_model, messages=messages)

    print(f"Result----------------------: {result.choices[0].message['content']}")
    return result.choices[0].message["content"]


def get_messages(context, question: str) -> List[Dict[str, str]]:
    context_str = "\n\n".join([f"Body:\n{x}" for x in context])
    print(f"context_str------------: {context_str}")
    print("---------------------")
    return [
        {
            "role": "system",
            "content": """
    You will receive a question from the user and some context to help you answer the question.

    Evaluate the context and provide an answer if you can confidently answer the question.

    If you are unable to provide a confident response, kindly state that it is the case and explain the reason.

    Prioritize offering an "I don't know" response over conveying potentially false information.

    The user will only see your response and not the context you've been provided. Thus, respond in precise detail, directly repeating the information that you're referencing from the context.
    """.strip(),
        },
        {
            "role": "user",
            "content": f"""
    Using the following information as context, I'd like you to answer a question.

    {context_str}

    Please answer the following question: {question}
    """.strip(),
        },
    ]


def main():
    embedding_cache = load_embedding_cache()
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
            plot, embedding_cache, model="text-embedding-ada-002"
        )

    # first_key, first_value = list(plot_embedding.items())[0]
    # print("First plot:", first_key)
    # print("First embedding:", first_value)
    # ask_embedding_store(
    #     "Do you know about any movies that have to do with heartbreak? If so can you tell me the plot?",
    #     plot_embedding,
    #     15,
    # )


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
