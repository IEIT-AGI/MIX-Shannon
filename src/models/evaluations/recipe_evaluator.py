import random
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Dict, Tuple
from omegaconf import DictConfig
from functools import partial


def compute_retrieval_metric(queries: np.ndarray,
                             database: np.ndarray,
                             distance_metric: str = 'cosine',
                             recall: Tuple[int] = (1, 5, 10)) -> Dict:
    """Function to compute Median Rank and Recall@k metrics given two sets of
       aligned embeddings.

    Args:
        queries (numpy.ndarray): A NxD dimensional array containing query
                                 embeddings.
        database (numpy.ndarray): A NxD dimensional array containing
                                  database embeddings.
        distance_metric (str): The distance metric to use to compare embeddings.
        recall (list): A list of integers with the k-values to
                             compute recall at.

    Returns:
        metrics (dict): A dictionary with computed values for each metric.
    """
    assert queries.shape == database.shape, "queries and database must have the same shape"
    assert len(recall) > 0, "recall cannot be empty"

    # largest k to compute recall
    max_k = int(max(recall))

    assert all(i >= 1 for i in recall), "all values in recall must be at least 1"
    assert max_k <= queries.shape[0], "the highest element in recall must be lower than database.shape[0]"

    dists = pairwise_distances(queries, database, metric=distance_metric)

    # find the number of elements in the ranking that have a lower distance
    # than the positive element (whose distance is in the diagonal
    # of the distance matrix) wrt the query. this gives the rank for each
    # query. (+1 for 1-based indexing)
    positions = np.count_nonzero(dists < np.diag(dists)[:, None], axis=-1) + 1

    # get the topk elements for each query (topk elements with lower dist)
    rankings = np.argpartition(dists, range(max_k), axis=-1)[:, :max_k]

    # positive positions for each query (inputs are assumed to be aligned)
    positive_idxs = np.array(range(dists.shape[0]))
    # matrix containing a cumulative sum of topk matches for each query
    # if cum_matches_topk[q][k] = 1, it means that the positive for query q
    # was already found in position <=k. if not, the value at that position
    # will be 0.
    cum_matches_top_k = np.cumsum(rankings == positive_idxs[:, None], axis=-1)
    # pre-compute all possible recall values up to k
    recall_values = np.mean(cum_matches_top_k, axis=0)

    # output
    metric = {}
    metric['median_recall'] = np.median(positions)
    for level in recall:
        metric[f'recall_{level}'] = recall_values[level - 1]
    return metric


def retrieval_evaluator(query_feats: np.ndarray,
                        database_feats: np.ndarray,
                        ranking_size: int,
                        run_eval_nums: int,
                        is_order_pick: bool = False,
                        distance_type: str = "euclidean") -> Dict:
    """Computes retrieval metrics for two sets of features

    Parameters
    ----------
    query_feats : np.ndarray [n x d]
        The image/recipe features..
    database_feats : np.ndarray [n x d]
        The recipe/image features.
    ranking_size : int
        Ranking size.
    run_eval_nums : int
        Number of evaluations to run (function returns the average).
    is_order_pick : bool
        Whether to force a particular order instead of picking random samples

    Returns
    -------
    dict
        Dictionary with metric values for all run_eval_nums runs.

    """
    metrics = {}
    for i in range(run_eval_nums):
        if is_order_pick:
            # pick the same samples in the same order for evaluation
            # is_order is only True when the function is used during training
            sub_ids = np.array(range(i * ranking_size, (i + 1) * ranking_size))
        else:
            sub_ids = random.sample(range(0, len(query_feats)), ranking_size)

        sub_query_feats = query_feats[sub_ids, :]
        sub_database_feats = database_feats[sub_ids, :]
        metric = compute_retrieval_metric(sub_query_feats, sub_database_feats, distance_metric=distance_type)
        for key, val in metric.items():
            if key in metrics:
                metrics[key].append(val)
            else:
                metrics[key] = [val]

    return metrics


class RecipeRetrievalEvaluator:
    METRIC_TYPE = {"retrieval_evaluator": retrieval_evaluator}

    def __init__(self, cfg: DictConfig) -> None:
        self.metric_names = []
        for metric_name, args in cfg.items():
            metric_fn = self.METRIC_TYPE.get(args.pop("type", "retrieval_evaluator"))
            setattr(self, metric_name, partial(metric_fn, **args))
            self.metric_names.append(metric_name)

    def __call__(self, csi_feats: np.ndarray, recipe_feats: np.ndarray) -> Dict:
        metrics = {}
        for metric_name in self.metric_names:
            fn = getattr(self, metric_name)
            metric = fn(recipe_feats, csi_feats) if metric_name.startswith("text") else fn(csi_feats, recipe_feats)
            metrics[metric_name] = metric

        return self.average_run_eval_nums(metrics)

    @staticmethod
    def average_run_eval_nums(metrics: Dict) -> Dict:
        rst = {}
        for name, val in metrics.items():
            rst[name] = {k: sum(v) / len(v) for k, v in val.items()}
        return rst
