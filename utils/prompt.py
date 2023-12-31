import tiktoken


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model: str) -> int:
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


def get_messages(context, question: str, chat_history):
    context_list = []

    for x in context:
        body_str = f"Movie:{x}\n"

        context_list.append(body_str)

    context_str = "\n\n".join(context_list)

    # print(f"context_str------------: {context_str}")
    # print("---------------------")
    new_array = [
        {
            "role": "system",
            "content": """
    You will receive a question from the user and some context to help you answer the question.

    Evaluate the context and  to determine if the question is related to the context. If so, use context to provide an answer if you can. If the context doesn't have the data needed to give a sensible response, provide a response that indicates that you don't have the data.

    If the question isn't related to the context, provide an answer based on the question alone.

    The user will only see your response and not the context you've been provided. Thus, respond in precise detail, directly repeating the information that you're referencing from the context.
    
    This is an ongoing conversation with the user, so you can use information from previous turns in the conversation to inform your response.
    """.strip(),
        },
        {
            "role": "user",
            "content": f"""
    Using the following information as context, I'd like you to answer a question.

    {context_str}

    Please respond to this statement or question from the user: {question}
    """.strip(),
        },
    ]

    # Insert 'messages' before the last item of 'new_array'
    new_array = new_array[:-1] + chat_history + new_array[-1:]

    return new_array
