import numpy as np
from config import config
from loguru import logger
from matplotlib import pyplot as plt


class Metrics:
    def __init__(self, recall_cutoff=config['METRICS_REPORT']):
        self._metrics = {}
        for k in recall_cutoff:
            self._metrics[f'HR_{k}'] = [[], hitrate]
            self._metrics[f'Recall_{k}'] = [[], recall]
            self._metrics[f'Precision_{k}'] = [[], precision]
            self._metrics[f'NDCG_{k}'] = [[], ndcg]

    def __call__(self, batch, x_pred):
        for key in self._metrics:
            func = self._metrics.get(key)[1]
            k = int(key.split('_')[1])
            preds = (-x_pred).argsort()[:, :k].numpy()
            ground_truth = batch.argsort()[:, -1].unsqueeze(1).numpy()
            self._metrics[key][0].append(func(
                preds, ground_truth, k))

    def log(self):
        res_string = []
        max_length = max(len(x) for x in self._metrics)
        res_dict = {}
        for metric in sorted(self._metrics, key=lambda x: (len(x), x)):
            metric_value = self._metrics.get(metric)
            logger.info(f"{metric.ljust(max_length)} | {np.mean(metric_value[0]):.8f}")
            res_dict[metric.ljust(max_length)] = np.mean(metric_value[0])

        return res_dict


def get_hr(preds, target_item, k=10):
    k = int(k)
    preds = np.array(preds)  # (batch_size, num_items)
    row = np.array(target_item)  # (batch_size, num_items)

    pred_items = (-preds).argsort().argsort()  # (batch_size, num_items) - ranks
    pred_items += 1
    pred_items[pred_items > k] = 0

    intersection = np.einsum('bd,bd->bd', pred_items, row)

    intersection[intersection > 0] = 1
    intersection = intersection.sum(1)

    return intersection.mean()


def plot_res_dict(res_dict):
    idx_metric = config['METRICS_REPORT']

    fig, ax = plt.subplots(1, len(idx_metric), figsize=(15, 5))

    for key in res_dict.keys():
        metric_values = np.array(res_dict[key])

        for axis in range(metric_values.shape[1]):
            ax[axis].plot(np.arange(metric_values.shape[0]),
                          metric_values[:, axis],
                          label=f'{key}')

            ax[axis].legend()
            ax[axis].set_title(f'HR_{idx_metric[axis]}')

    plt.show()


def user_hitrate(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single hitrate
    """
    return len(set(rank[:k]).intersection(set(ground_truth)))


def hitrate(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_hitrate(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()


def user_precision(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single precision
    """
    return user_hitrate(rank, ground_truth, k) / len(rank[:k])


def precision(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_precision(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()


def user_recall(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single recall
    """
    return user_hitrate(rank, ground_truth, k) / len(ground_truth)


def recall(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_recall(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()


def user_ap(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single ap
    """
    return np.sum([
        user_precision(rank, ground_truth, idx + 1)
        for idx, item in enumerate(rank[:k]) if item in ground_truth
    ]) / len(rank[:k])


def ap(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_ap(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()


def map(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: single map
    """
    return np.mean([ap(rank, ground_truth, k)])


def user_ndcg(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single ndcg
    """
    dcg = 0
    idcg = 0
    for idx, item in enumerate(rank[:k]):
        dcg += 1.0 / np.log2(idx + 2) if item in ground_truth else 0.0
        idcg += 1.0 / np.log2(idx + 2)
    return dcg / idcg


def ndcg(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_ndcg(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()


def user_mrr(rank, ground_truth, k=20):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single mrr
    """
    for idx, item in enumerate(rank[:k]):
        if item in ground_truth:
            return 1 / (idx + 1)
    return 0


def mrr(rank, ground_truth, k=20):
    """
    :param rank: shape [n_users, n_recommended_items]
    :param ground_truth: shape [n_users, n_relevant_items]
    :param k: number of top recommended items
    :return: shape [n_users]
    """
    return np.array([
        user_mrr(user_rank, user_ground_truth, k)
        for user_rank, user_ground_truth in zip(rank, ground_truth)
    ]).mean()
