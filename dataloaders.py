import numpy as np
import pandas as pd
from geo import GeoDist
from config import config
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatasetReader:

    def __init__(self,
                 path,
                 nrows=1000,
                 sep='\t',
                 header=None,
                 geo_precision=2,
                 names=('userId', 'timestamp', 'long', 'lat', 'loc_id')):
        self.config = config
        self.df = pd.read_csv(path,
                              nrows=nrows,
                              sep=sep,
                              header=header,
                              names=names)

        self.geo_p = geo_precision

    def get_dataloaders(self,
                        test_size=0.3):
        self._filter()
        self.user_encoder, self.loc_id_encoder, self.geo_encoder = self._get_encoders()
        self._encode()
        self.df = self._get_groups()

        train_df, test_df = train_test_split(self.df,
                                             test_size=test_size)
        return train_df, test_df

    def _filter(self):
        # self.df.sort_values(by='timestamp',ascending=True,inplace=True)
        self.df.drop_duplicates(subset=['userId', 'loc_id'], keep='first', inplace=True)
        # print(self.df.shape)
        # GroupBy item
        temp = self.df.groupby('loc_id').agg({'userId': 'count'}).reset_index()
        # Treshold item ratings
        items = temp.loc[temp.userId > config['NUM_RAT_FOR_ITEM']].loc_id
        self.df = self.df.loc[self.df.loc_id.isin(items)].copy()
        # GroupBy user
        temp = self.df.groupby('userId').agg({'loc_id': 'count'}).reset_index()
        # Threshold user ratings
        # print(temp)
        users = temp.loc[temp.loc_id > config['NUM_RAT_FOR_USER']].userId
        # print(users)
        self.df = self.df.loc[self.df.userId.isin(users)].copy()

    def _get_encoders(self):
        user_encoder = {v: k for k, v in enumerate(self.df.userId.unique())}
        loc_id_encoder = {v: k for k, v in enumerate(self.df.loc_id.unique())}

        self.df['geo_place'] = self.df.apply(
            lambda x: f'{round(x["long"], self.geo_p)}_{round(x["lat"], self.geo_p)}',
            axis=1
        )

        geo_encoder = {v: k for k, v in enumerate(self.df.geo_place.unique())}

        print(f'Количество юзеров: {len(user_encoder)}\nКоличество айтемов: {len(loc_id_encoder)}')
        print(f'Количество гео-позиций: {len(geo_encoder)}')
        print(f'Количество интеракций: {self.df.shape[0]}')
        return user_encoder, loc_id_encoder, geo_encoder

    def _encode(self):
        self.df.userId = self.df.userId.apply(lambda x: self.user_encoder[x])
        self.df.loc_id = self.df.loc_id.apply(lambda x: self.loc_id_encoder[x])
        self.df.geo_place = self.df.geo_place.apply(lambda x: self.geo_encoder[x])

    def _get_groups(self):
        data_grouped = self.df.groupby('userId').apply(
            lambda x: sorted(zip(x.loc_id, x.timestamp, x.long, x.lat, x.geo_place),
                             key=lambda x: x[1])
        ).reset_index()

        data_grouped['items'] = data_grouped[0].apply(lambda x: [inter[0] for inter in x])
        data_grouped['timestamp'] = data_grouped[0].apply(
            lambda x: [pd.to_datetime(inter[1]).timestamp() for inter in x])
        data_grouped['long'] = data_grouped[0].apply(lambda x: [inter[2] for inter in x])
        data_grouped['lat'] = data_grouped[0].apply(lambda x: [inter[3] for inter in x])
        data_grouped['geo'] = data_grouped[0].apply(lambda x: [(inter[2], inter[3]) for inter in x])
        data_grouped['geo_idx'] = data_grouped[0].apply(lambda x: [inter[4] for inter in x])
        data_grouped.drop([0], axis=1, inplace=True)

        return data_grouped


class MyDataset(Dataset):
    def __init__(self,
                 df,
                 num_items,
                 max_length=config['MAX_LENGTH'],
                 max_time_span: int = config['MAX_TIME_SPAN'],
                 time_unit: str = 'h'):

        if time_unit not in ['s', 'm', 'h', 'd']:
            raise ValueError(f'unknown time unit value {time_unit}')
        self._df = df
        self._max_length = max_length
        self._all_items = np.arange(num_items)
        self._num_items = num_items
        self._max_time_span = max_time_span
        self._time_unit = time_unit

        self.geo_dist = GeoDist()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        interactions = row['items']
        length = len(interactions[:-2])
        pos_items = interactions[:-2]
        # Предсказываем следующий элемент, надо сдвинуть на 1 поэтому
        target = interactions[1:-1]

        distances = self.geo_dist.geoseq2distseq(row['geo'][:-2])
        geo = row['geo_idx'][:-2]

        ''' Этот элемент будет для валидации'''
        valid_elem = np.zeros(self._num_items)
        valid_elem[interactions[-1]] = 1

        if length > self._max_length:
            pos_items = pos_items[-self._max_length:]
            distances = distances[-self._max_length:]
            target = target[-self._max_length:]
            geo = geo[-self._max_length:]

        return (pos_items,
                distances,
                geo,
                target,
                valid_elem)
