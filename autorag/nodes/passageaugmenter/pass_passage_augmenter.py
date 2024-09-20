from typing import List

import pandas as pd

from autorag.nodes.passageaugmenter.base import BasePassageAugmenter
from autorag.utils import result_to_dataframe


class PassPassageAugmenter(BasePassageAugmenter):
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		top_k = kwargs.pop("top_k")

		_, ids = self.cast_to_run(previous_result)
		contents = previous_result["retrieved_contents"].tolist()
		scores = previous_result["retrieve_scores"].tolist()

		augmented_ids, augmented_contents, augmented_scores = self._pure(
			ids, contents, scores
		)
		return self.sort_by_scores(
			augmented_contents, augmented_ids, augmented_scores, top_k
		)

	def _pure(
		self,
		ids_list: List[List[str]],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
	):
		"""
		Do not perform augmentation.
		Return given passages, scores, and ids as is.
		"""
		return ids_list, contents_list, scores_list
