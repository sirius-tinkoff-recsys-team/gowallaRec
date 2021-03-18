import numpy as np
from config import config
from loguru import logger
from matplotlib import pyplot as plt

class Metrics:
    def __init__(self, recall_cutoff=config['METRICS_REPORT']):
        self._metrics = {'HR_{}'.format(k): [[], get_hr] for k in recall_cutoff}

    def __call__(self, x_pred, batch):
        for key in self._metrics:
            func = self._metrics.get(key)[1]
            self._metrics[key][0].append(func(x_pred, batch, key.split('_')[1]))

    def log(self):
        res_string = []
        max_length = max(len(x) for x in self._metrics)
        res_dict = {}
        for metric in sorted(self._metrics, key=lambda x: (len(x), x)):
            metric_value = self._metrics.get(metric)
            logger.info(f"{metric.ljust(max_length)} | {np.mean(metric_value[0]):.3f}")
            res_dict[metric.ljust(max_length)] = np.mean(metric_value[0])

        return res_dict


def get_hr(target_item, preds, k=10):
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