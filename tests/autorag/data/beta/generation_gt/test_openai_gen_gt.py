import time
from unittest.mock import patch

import openai.resources.beta.chat
from openai import AsyncOpenAI
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChoice,
	ParsedChatCompletionMessage,
)

from autorag.data.beta.generation_gt.openai_gen_gt import (
	make_concise_gen_gt,
	make_basic_gen_gt,
	Response,
)
from autorag.schema.data import QA
from tests.autorag.data.beta.generation_gt.base_test_generation_gt import (
	qa_df,
	check_generation_gt,
)

client = AsyncOpenAI()


async def mock_gen_gt_response(*args, **kwargs) -> ParsedChatCompletion[Response]:
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=Response(answer="mock answer"),
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model="gpt-4o-mini-2024-07-18",
		object="chat.completion",
	)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_gen_gt_response,
)
def test_make_concise_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(
		lambda row: make_concise_gen_gt(
			row, client, model_name="gpt-4o-mini-2024-07-18"
		)
	)
	check_generation_gt(result_qa)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_gen_gt_response,
)
def test_make_basic_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(lambda row: make_basic_gen_gt(row, client))
	check_generation_gt(result_qa)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_gen_gt_response,
)
def test_make_basic_gen_gt_ko():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(lambda row: make_basic_gen_gt(row, client, lang="ko"))
	check_generation_gt(result_qa)
