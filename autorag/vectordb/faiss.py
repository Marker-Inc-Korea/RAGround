import logging


import faiss

from typing import List, Tuple

from autorag.utils.util import apply_recursive
from autorag.vectordb import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class Faiss(BaseVectorStore):
	def __init__(
		self,
		embedding_model: str,
		index_name: str,
		embedding_batch: int = 100,
		similarity_metric: str = "cosine",
		dimension: int = 768,
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)

		self.index = faiss.IndexFlatL2(dimension)

	async def add(self, ids: List[str], texts: List[str]):
		texts = self.truncated_inputs(texts)
		text_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(texts)

		return text_embeddings

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		results = self.index.fetch(ids=ids, namespace=self.namespace)
		id_vector_dict = {
			str(key): val["values"] for key, val in results["vectors"].items()
		}
		result = [id_vector_dict[_id] for _id in ids]
		return result

	async def is_exist(self, ids: List[str]) -> List[bool]:
		fetched_result = self.index.fetch(ids=ids, namespace=self.namespace)
		existed_ids = list(map(str, fetched_result.get("vectors", {}).keys()))
		return list(map(lambda x: x in existed_ids, ids))

	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		queries = self.truncated_inputs(queries)
		query_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(queries)

		ids, scores = [], []
		for query_embedding in query_embeddings:
			response = self.index.query(
				vector=query_embedding,
				top_k=top_k,
				include_values=True,
				namespace=self.namespace,
			)

			ids.append([o.id for o in response.matches])
			scores.append([o.score for o in response.matches])

		if self.similarity_metric in ["l2"]:
			scores = apply_recursive(lambda x: -x, scores)

		return ids, scores

	async def delete(self, ids: List[str]):
		# Delete entries by IDs
		self.index.delete(ids=ids, namespace=self.namespace)

	def delete_index(self):
		# Delete the index
		self.client.delete_index(self.index_name)
