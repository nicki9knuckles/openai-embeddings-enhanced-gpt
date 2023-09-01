import openai
import tiktoken
import os
from dotenv import load_dotenv
import argparse


from utils.embeddings import ask_embedding_store
from utils.get_movie_embeddings import get_movie_embeddings


MAX_CONTEXT_WINDOW = 4097
MINIMUM_RESPONSE_SPACE = 1000
MAX_PROMPT_SIZE = MAX_CONTEXT_WINDOW - MINIMUM_RESPONSE_SPACE
chat_model = "gpt-3.5-turbo"
embedding_enc = tiktoken.encoding_for_model("text-embedding-ada-002")
enc = tiktoken.encoding_for_model(chat_model)


def bold(text):
    bold_start = "\033[1m"
    bold_end = "\033[0m"
    return bold_start + text + bold_end


def blue(text):
    blue_start = "\033[34m"
    blue_end = "\033[0m"
    return blue_start + text + blue_end


def red(text):
    red_start = "\033[31m"
    red_end = "\033[0m"
    return red_start + text + red_end


def main():
    movie_embeddings = get_movie_embeddings()
    parser = argparse.ArgumentParser(description="Simple command line chatbot")
    parser.add_argument(
        "--personality",
        type=str,
        help="A brief summary of the chatbot's personality",
        default="Friendly and helpful",
    )
    args = parser.parse_args()

    initial_prompt = (
        f"You are a movie recommendation chatbot. Your personality is: {args}"
    )
    messages = [
        {
            "role": "system",
            "content": initial_prompt,
        }
    ]

    while True:
        try:
            user_input = input(bold(blue("You: ")))
            messages.append({"role": "user", "content": user_input})

            res = ask_embedding_store(
                MAX_PROMPT_SIZE,
                chat_model,
                enc,
                messages,
                movie_embeddings,
                10,
            )

            messages.append(res["choices"][0]["message"].to_dict())
            print(bold(red("Assistant: ")), res["choices"][0]["message"]["content"])
        except KeyboardInterrupt:
            print("Exiting...")
            break


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
