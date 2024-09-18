import abc
import logging
from pathlib import Path
from typing import List, Union, Dict, Optional

import pandas as pd

from autorag.schema import BaseModule
from autorag.support import get_support_modules
from autorag.utils import validate_qa_dataset

logger = logging.getLogger("AutoRAG")


class BaseQueryExpansion(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		self.cast_to_init(project_dir, *args, **kwargs)

	def cast_to_init(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize query expansion node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		validate_qa_dataset(previous_result)

		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		# pop prompt from kwargs
		prompt = kwargs.pop("prompt", "")

		# set generator module for query expansion
		generator_callable, generator_param = make_generator_callable_param(kwargs)
		return queries, prompt, generator_callable, generator_param

	@staticmethod
	def _check_expanded_query(queries: List[str], expanded_queries: List[List[str]]):
		return list(
			map(
				lambda query, expanded_query_list: check_expanded_query(
					query, expanded_query_list
				),
				queries,
				expanded_queries,
			)
		)


def check_expanded_query(query: str, expanded_query_list: List[str]):
	# check if the expanded query is the same as the original query
	expanded_query_list = list(map(lambda x: x.strip(), expanded_query_list))
	return [
		expanded_query if expanded_query else query
		for expanded_query in expanded_query_list
	]


def make_generator_callable_param(generator_dict: Optional[Dict]):
	if "generator_module_type" not in generator_dict.keys():
		generator_dict = {
			"generator_module_type": "llama_index_llm",
			"llm": "openai",
			"model": "gpt-4o-mini",
		}
	module_str = generator_dict.pop("generator_module_type")
	module_callable = get_support_modules(module_str)
	module_param = generator_dict
	return module_callable, module_param
