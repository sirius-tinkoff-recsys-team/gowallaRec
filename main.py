from config import config
from models import LSTMModel
from utils import custom_collate
from metrics import plot_res_dict
from torch.utils.data import DataLoader
from dataloaders import DatasetReader, MyDataset

if __name__ == '__main__':
    exp_results = {}
    epochs = 20

    reader = DatasetReader('loc-brightkite_totalCheckins.txt.gz', nrows=None)
    train_ds, test_ds = reader.get_dataloaders()

    train_loader = DataLoader(
        MyDataset(train_ds, num_items=len(reader.loc_id_encoder)),
        batch_size=config['TRAIN_BATCH_SIZE'],
        shuffle=False,
        collate_fn=custom_collate
    )

    test_loader = DataLoader(
        MyDataset(test_ds, num_items=len(reader.loc_id_encoder)),
        batch_size=config['TEST_BATCH_SIZE'],
        shuffle=False,
        collate_fn=custom_collate
    )

    model = LSTMModel(num_items=len(reader.loc_id_encoder),
                      num_geo=len(reader.geo_encoder),
                      use_geo=False, device='cpu')

    model.set_optimizer()

    model.validate(test_loader)

    epochs_metric = []

    for epoch in range(epochs):
        model.fit(train_loader)
        epoch_metric = model.validate(test_loader)
        epochs_metric.append([epoch_metric[k] for k in epoch_metric.keys()])

    exp_results['lstm_no_geo'] = epochs_metric
    plot_res_dict(exp_results)
