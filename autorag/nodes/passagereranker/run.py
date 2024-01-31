import logging
import os
import pathlib
from typing import List, Callable, Dict

import pandas as pd

from autorag.nodes.retrieval.run import evaluate_retrieval_node
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name

logger = logging.getLogger("AutoRAG")


def run_passage_reranker_node(modules: List[Callable],
                       module_params: List[Dict],
                       previous_result: pd.DataFrame,
                       node_line_dir: str,
                       strategies: Dict,
                       ) -> pd.DataFrame:
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # run metrics before filtering
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for passage_reranker evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), results))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "passage_reranker")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepaths = list(map(lambda x: os.path.join(save_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
        **{f'passage_reranker_{metric}': list(map(lambda result: result[metric].mean(), results)) for metric in
           strategies.get('metrics')},
    })

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best_average(results, strategies.get('metrics'), filenames)
    best_result = pd.concat([previous_result, selected_result], axis=1)
    best_result = best_result.drop(columns=['retrieved_contents', 'retrieved_ids', 'retrieve_scores'])

    # add summary.parquet 'is_best' column
    summary_df['is_best'] = summary_df['filename'] == selected_filename

    # save files
    summary_df.to_parquet(os.path.join(save_dir, "summary.parquet"), index=False)
    best_result.to_parquet(os.path.join(save_dir, f'best_{os.path.splitext(selected_filename)[0]}.parquet'), index=False)
    return best_result
