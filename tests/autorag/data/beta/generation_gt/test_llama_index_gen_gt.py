from llama_index.core.llms import MockLLM

from autorag.data.beta.generation_gt.llama_index_gen_gt import (
	make_concise_gen_gt,
	make_basic_gen_gt,
)
from autorag.data.beta.schema.data import QA
from tests.autorag.data.beta.generation_gt.base_test_generation_gt import (
	qa_df,
	check_generation_gt,
)

llm = MockLLM()


def test_make_concise_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_concise_gen_gt, llm=llm)
	check_generation_gt(result_qa)


def test_make_basic_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm)
	check_generation_gt(result_qa)


def test_make_basic_gen_gt_ko():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm, lang="ko")
	check_generation_gt(result_qa)
