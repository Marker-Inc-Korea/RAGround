import os
import tempfile
from datetime import datetime

import chromadb
import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval.bm25 import bm25_ingest
from autorag.nodes.retrieval.vectordb import vectordb_ingest

sample_ids = ([
                  ['id-1', 'id-2', 'id-3'],
                  ['id-2', 'id-3', 'id-4']
              ], [
                  ['id-1', 'id-4', 'id-3'],
                  ['id-2', 'id-5', 'id-4']
              ])
sample_scores = ([
                     [1, 3, 5],
                     [2, 4, 6]
                 ], [
                     [4, 2, 6],
                     [5, 3, 7]
                 ])
sample_ids_2 = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'])
sample_scores_2 = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])

sample_ids_3 = ([
                    ['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                    ['id-2', 'id-3', 'id-4', 'id-5', 'id-6']
                ], [
                    ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'],
                    ['id-2', 'id-5', 'id-4', 'id-6', 'id-3']
                ])
sample_scores_3 = ([
                       [5, 3, 1, 0.4, 0.2],
                       [6, 4, 2, 1.4, 1.2]
                   ], [
                       [6, 2, 1, 0.5, 0.1],
                       [7, 3, 2, 1.5, 1.1],
                   ])
sample_retrieval_gt_3 = [
    [['id-3']],
    [['id-2'], ['id-6']],
]

sample_ids_non_overlap = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                          ['id-6', 'id-4', 'id-3', 'id-7', 'id-2'])

previous_result = pd.DataFrame({
    'qid': ['query-1', 'query-2', 'query-3'],
    'query': ['query-1', 'query-2', 'query-3'],
    'retrieval_gt': [
        [['id-1'], ['id-2'], ['id-3']],
        [['id-1'], ['id-2'], ['id-3']],
        [['id-1'], ['id-2'], ['id-3']]
    ],
    'generation_gt': [
        ['gen-1', 'gen-2'],
        ['gen-1', 'gen-2'],
        ['gen-1', 'gen-2']
    ]
})

modules_with_weights = {
    'ids': ([['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
             ['id-1', 'id-2', 'id-3', 'id-4', 'id-5']],
            [['id-1', 'id-4', 'id-3', 'id-5', 'id-2'],
             ['id-1', 'id-4', 'id-3', 'id-5', 'id-2']]
            ),
    'scores': ([[5, 3, 1, 0.4, 0.2], [5, 3, 1, 0.4, 0.2]],
               [[6, 2, 1, 0.5, 0.1], [6, 2, 1, 0.5, 0.1]]),
    'top_k': 3,
    'weight_range': (0.3, 0.7),
    'test_weight_size': 20,
}


@pytest.fixture
def pseudo_project_dir():
    with tempfile.TemporaryDirectory() as project_dir:
        corpus_df = pd.DataFrame({
            'doc_id': ['id-1', 'id-2', 'id-3', 'id-4', 'id-5', 'id-6', 'id-7', 'id-8', 'id-9'],
            'contents': ['doc-1', 'doc-2', 'doc-3', 'doc-4', 'doc-5', 'doc-6', 'doc-7', 'doc-8', 'doc-9'],
            'metadata': [{'last_modified_date': datetime.now()} for _ in range(9)]
        })
        qa_df = pd.DataFrame({
            'qid': ['havertz', 'hanjunsu'],
            'query': ['What is JAX?', 'Donggeon twerking jax havertz?'],
            'retrieval_gt': [
                [['id-1', 'id-4']],
                [['id-3', 'id-8'], ['id-2']],
            ],
            'generation_gt': [
                ['JAX is minsingjin.'],
                ['Donggeon is the god.'],
            ]
        })
        os.makedirs(os.path.join(project_dir, "data"))
        corpus_df.to_parquet(os.path.join(project_dir, "data", 'corpus.parquet'))
        qa_df.to_parquet(os.path.join(project_dir, "data", 'qa.parquet'))
        resource_dir = os.path.join(project_dir, "resources")
        os.makedirs(resource_dir)
        bm25_ingest(os.path.join(resource_dir, 'bm25_porter_stemmer.pkl'), corpus_df)
        chroma_path = os.path.join(resource_dir, 'chroma')
        db = chromadb.PersistentClient(path=chroma_path)
        collection = db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})
        vectordb_ingest(collection, corpus_df, OpenAIEmbedding())
        yield project_dir


def base_hybrid_weights_node_test(hybrid_func, pseudo_project_dir, retrieve_scores):
    result = hybrid_func(project_dir=pseudo_project_dir, previous_result=previous_result, **modules_with_weights)
    assert len(result) == 2
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'retrieved_contents', 'retrieved_ids', 'retrieve_scores'}
    assert result['retrieved_ids'].tolist()[0] == ['id-1', 'id-4', 'id-2']
    assert result['retrieve_scores'].tolist()[0] == pytest.approx(retrieve_scores)
    assert result['retrieved_contents'].tolist()[0] == ['doc-1', 'doc-4', 'doc-2']
