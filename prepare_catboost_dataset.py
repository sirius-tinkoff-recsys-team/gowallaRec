import utils
import implicit
import pandas as pd
from config import config
from model import LightGCN, TopNModel, TopNPersonalized, TopNNearestModel
from dataloader import GowallaLightGCNDataset, GowallaTopNDataset, GowallaALSDataset


def get_als_recommendations(path):
    gowalla_train, user_item_data, item_user_data = GowallaALSDataset(path) \
        .get_dataset(5739, 56261)  # n_users and m_items from list.txt

    model = implicit.als.AlternatingLeastSquares(iterations=3, factors=64)
    model.fit(item_user_data)

    users = []
    items_pred = []
    for user in range(5738 + 1):
        preds = list(map(lambda x: x[0], model.recommend(user, user_item_data, 20)))
        users.extend([user for _ in preds])
        items_pred.extend(preds)
    return users, items_pred


def get_topn_recommendations(path):
    train_dataset = GowallaTopNDataset(path)

    model = TopNPersonalized(15)
    model.fit(train_dataset)

    users = []
    items_pred = []
    for user in range(5738 + 1):
        preds = list(model.recommend([user])[0])
        users.extend([user for _ in preds])
        items_pred.extend(preds)
    return users, items_pred


if __name__ == '__main__':
    # make candidates for catboost training
    als_users, als_items = get_als_recommendations('dataset/gowalla.train')
    topn_users, topn_items = get_topn_recommendations('dataset/gowalla.train')
    pd.DataFrame({'userId': als_users + topn_users, 'itemId': als_items + topn_items}) \
        .to_csv('catboost_train_dataset.csv', index=None, header=None)

    # make candidates for catboost eval
    als_users, als_items = get_als_recommendations('dataset/gowalla.traintest')
    topn_users, topn_items = get_topn_recommendations('dataset/gowalla.traintest')
    pd.DataFrame({'userId': als_users + topn_users, 'itemId': als_items + topn_items}) \
        .to_csv('catboost_eval_dataset.csv', index=None, header=None)

    pd.DataFrame({'userId': als_users, 'itemId': als_items}) \
        .to_csv('als_candidates.csv', index=None, header=None)
    pd.DataFrame({'userId': topn_users, 'itemId': topn_items}) \
        .to_csv('topn_candidates.csv', index=None, header=None)

    # if config['MODEL'] == 'LightGCN':
    #     train_dataset = GowallaLightGCNDataset('dataset/gowalla.train')
    #     test_dataset = GowallaLightGCNDataset('dataset/gowalla.test', train=False)
    #
    #     model = LightGCN(train_dataset)
    #     model.fit(config['TRAIN_EPOCHS'], test_dataset)
    # elif config['MODEL'] == 'TopNModel':
    #     train_dataset = GowallaTopNDataset('dataset/gowalla.train')
    #     test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)
    #
    #     model = TopNModel(config['TOP_N'])
    #     model.fit(train_dataset)
    #     model.eval(test_dataset)
    # elif config['MODEL'] == 'TopNPersonalized':
    #     train_dataset = GowallaTopNDataset('dataset/gowalla.train')
    #     test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)
    #
    #     model = TopNPersonalized(config['TOP_N'])
    #     model.fit(train_dataset)
    #     model.eval(test_dataset)
    # elif config['MODEL'] == 'TopNNearestModel':
    #     train_dataset = GowallaTopNDataset('dataset/gowalla.train')
    #     test_dataset = GowallaTopNDataset('dataset/gowalla.test', train=False)
    #
    #     calc_nearest = utils.calc_nearest(train_dataset, test_dataset)
    #     model = TopNNearestModel(config['TOP_N'], calc_nearest)
    #     model.fit(train_dataset)
    #     model.eval(test_dataset)
    # elif config['MODEL'] == 'iALS':
    #     gowalla_train, user_item_data, item_user_data = GowallaALSDataset(
    #         'dataset/gowalla.train').get_dataset()
    #     gowalla_test = GowallaALSDataset('dataset/gowalla.test', train=False).get_dataset()
    #     model = implicit.als.AlternatingLeastSquares(
    #         iterations=config['ALS_N_ITERATIONS'], factors=config['ALS_N_FACTORS'])
    #     model.fit_callback = utils.eval_als_model(model, user_item_data, gowalla_test)
    #     model.fit(item_user_data)
