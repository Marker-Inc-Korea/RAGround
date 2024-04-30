import asyncio
import logging
import os
from typing import List, Tuple

import tiktoken
from openai import AsyncOpenAI
from tiktoken import Encoding

from autorag.nodes.generator.base import generator_node
from autorag.utils.util import process_batch

logger = logging.getLogger("AutoRAG")


@generator_node
def openai(prompts: List[str], model: str = "gpt-3.5-turbo", batch: int = 16,
           truncate: bool = True,
           api_key: str = None,
           **kwargs) -> \
        Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    OpenAI generator module.
    Uses official openai library for generating answer from the given prompt.
    It returns real token ids and log probs, so you must use this for using token ids and log probs.

    :param prompts: A list of prompts.
    :param model: A model name for openai.
        Default is gpt-3.5-turbo.
    :param batch: Batch size for openai api call.
        If you get API limit errors, you should lower the batch size.
        Default is 16.
    :param truncate: Whether to truncate the input prompt.
        Default is True.
    :param api_key: OpenAI API key. You can set this by passing env variable `OPENAI_API_KEY`
    :param kwargs: The optional parameter for openai api call `openai.chat.completion`
        See https://platform.openai.com/docs/api-reference/chat/create for more details.
    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids.
        The third element is a list of generated text's log probs.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY does not set. "
                             "Please set env variable OPENAI_API_KEY or pass api_key parameter to openai module.")

    client = AsyncOpenAI(api_key=api_key)
    tokenizer = tiktoken.encoding_for_model(model)
    loop = asyncio.get_event_loop()
    tasks = [get_result(prompt, client, model, tokenizer, **kwargs) for prompt in prompts]
    result = loop.run_until_complete(process_batch(tasks, batch))
    answer_result = list(map(lambda x: x[0], result))
    token_result = list(map(lambda x: x[1], result))
    logprob_result = list(map(lambda x: x[2], result))
    return answer_result, token_result, logprob_result


async def get_result(prompt: str, client: AsyncOpenAI, model: str, tokenizer: Encoding, **kwargs):
    if kwargs.pop('logprobs') is not None:
        logger.warning("parameter logprob does not effective. It always set to True.")
    if kwargs.pop('n') is not None:
        logger.warning("parameter n does not effective. It always set to 1.")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
        n=1,
        **kwargs
    )
    choice = response.choices[0]
    answer = choice.message.content
    logprobs = list(map(lambda x: x['logprob'], choice.logprobs.content))
    tokens = tokenizer.encode(choice)
    assert len(tokens) == len(logprobs), "tokens and logprobs size is different."
    return answer, tokens, logprobs
