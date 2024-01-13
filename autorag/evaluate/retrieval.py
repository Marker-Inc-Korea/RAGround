import functools
import warnings
from typing import List, Callable, Any, Tuple

import pandas as pd

from autorag.evaluate.metric import retrieval_recall, retrieval_precision, retrieval_f1


def evaluate_retrieval(retrieval_gt: List[List[List[str]]], metrics: List[str]):
    def decorator_evaluate_retrieval(
            func: Callable[[Any], Tuple[List[List[str]], List[List[float]], List[List[str]]]]):
        """
        Decorator for evaluating retrieval results.
        You can use this decorator to any method that returns (contents, scores, retrieval_gt),
        which is the output of conventional retrieval modules.

        :param func: Must return (contents, scores, retrieval_gt)
        :return: wrapper function that returns pd.DataFrame, which is the evaluation result.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            contents, scores, pred_ids = func(*args, **kwargs)
            metric_funcs = {
                'recall': retrieval_recall,
                'precision': retrieval_precision,
                'f1': retrieval_f1,
            }

            metric_scores = {}
            for metric in metrics:
                if metric not in metric_funcs:
                    warnings.warn(f"metric {metric} is not in supported metrics: {metric_funcs.keys()}"
                                  f"{metric} will be ignored.")
                metric_func = metric_funcs[metric]
                metric_scores[metric] = metric_func(retrieval_gt=retrieval_gt, ids=pred_ids)

            metric_result_df = pd.DataFrame(metric_scores)
            execution_result_df = pd.DataFrame({
                'contents': contents,
                'scores': scores,
                'pred_ids': pred_ids,
                'retrieval_gt': retrieval_gt,
            })
            result_df = pd.concat([execution_result_df, metric_result_df], axis=1)
            return result_df

        return wrapper

    return decorator_evaluate_retrieval
