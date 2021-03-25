import metrics
import faiss
import torch
from loguru import logger
import numpy as np
from time import gmtime, strftime
import pandas as pd

if __name__ == '__main__':
    gowalla_train = pd.read_csv('dataset/gowalla.train',
                                names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
    gowalla_test = pd.read_csv('dataset/gowalla.test',
                               names=['userId', 'timestamp', 'long', 'lat', 'loc_id'])
    gowalla_dataset = pd.concat([gowalla_train, gowalla_test])
    print(gowalla_dataset.columns)
    print(gowalla_dataset['lat'])

    # current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    # logger.add(f'train_{current_time}.log')
    # for i in range(5):
    #     logger.info(f'i = {i}')

    # d = 64
    # embedding_user = torch.nn.Embedding(
    #     num_embeddings=10000, embedding_dim=d)
    # embedding_item = torch.nn.Embedding(
    #     num_embeddings=10000, embedding_dim=d)
    #
    # user_emb = embedding_user.weight.detach().numpy()
    # items_emb = embedding_item.weight.detach().numpy()
    #
    # index = faiss.IndexHNSWPQ(d, 4, 32)
    # print(index.is_trained)
    # index.train(items_emb)
    # index.add(items_emb)
    # print(index.ntotal)
    # print(index.search(user_emb, 20)[1])

    # index = faiss.IndexHNSWPQ(d, faiss.ScalarQuantizer.QT_8bit, 32)
    # index.add(xb)
    # print(index.search(xq, 20))

    # k = 1
    # rank = [[1, 2, 3]]
    # ground_truth = [[3, 4, 5]]
    # print(f'rank: {rank}, ground_truth: {ground_truth}, ap@{k} = {metrics.ap(rank, ground_truth, k=k)}')
    #
    # rank = [[1, 2, 3]]
    # ground_truth = [[1, 4, 5]]
    # print(f'rank: {rank}, ground_truth: {ground_truth}, ap@{k} = {metrics.ap(rank, ground_truth, k=k)}')
    #
    # rank = [[1, 2, 3]]
    # ground_truth = [[1, 2, 3]]
    # print(f'rank: {rank}, ground_truth: {ground_truth}, ap@{k} = {metrics.ap(rank, ground_truth, k=k)}')
    #
    # rank = [[1, 2, 3]]
    # ground_truth = [[1, 2, 3, 4, 5]]
    # print(f'rank: {rank}, ground_truth: {ground_truth}, ap@{k} = {metrics.ap(rank, ground_truth, k=k)}')

    # rank = [[2, 3, 1], [8, 4, 1]]
    # ground_truth = [[1, 2, 3], [1, 4, 3]]
    # for k in range(1, 11):
    #     print(f'hitrate@{k} = {metrics.hitrate(rank, ground_truth, k=k)}')
    #     print(f'precision@{k} = {metrics.precision(rank, ground_truth, k=k)}')
    #     print(f'recall@{k} = {metrics.recall(rank, ground_truth, k=k)}')
    #     print(f'ap@{k} = {metrics.ap(rank, ground_truth, k=k)}')
    #     print(f'map@{k} = {metrics.map(rank, ground_truth, k=k)}')
    #     print(f'ndcg@{k} = {metrics.ndcg(rank, ground_truth, k=k)}')
    #     print(f'mrr@{k} = {metrics.mrr(rank, ground_truth, k=k)}')
    #     print('-----')
