import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from metrics import Metrics


class BaseModel(torch.nn.Module):
    def __init__(self,
                 num_items,
                 hidden_dim: int = 64,
                 max_time_span: int = 64,
                 device='cpu'):

        super().__init__()

        self._current_count = 0

        self._item_embedding = torch.nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=hidden_dim,
            padding_idx=0
        )

        self._dense_decoder = torch.nn.Linear(hidden_dim, num_items)
        self._optimizer = None
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.to(device)
        self.device = device

    def set_optimizer(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, inp):

        user_timestamp = self.encode(inp)
        decoded = self._dense_decoder(user_timestamp)
        return decoded

    def encode(self, inp):
        raise NotImplementedError

    def fit(self, loader):
        self.train()
        iterator = tqdm(loader)
        gen_loss = []
        for batch in iterator:
            # batch['items'] = batch['items'].to(self.device)
            for k in batch:
                batch[k] = batch[k].to(self.device)
            output = self(batch)
            target = batch['target']
            loss = self.get_loss(output, target)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            gen_loss.append(loss.item())
            mean_cur_loss = np.mean(gen_loss)
            iterator.set_description('Train phase, Loss: {:.2f}'.format(np.mean(gen_loss)))

    def get_loss(self, output, target):

        output = output.view(output.shape[0] * output.shape[1], output.shape[2])
        target = target.view(target.shape[0] * target.shape[1]).long()

        loss = self.criterion(output, target)
        return loss

    @torch.no_grad()
    def validate(self, loader):
        self.eval()
        iterator = tqdm(loader)
        metrics = Metrics()
        gen_loss = []
        for batch in iterator:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            output = self(batch).detach().cpu()  # (batch_size, num_items)
            valid_elem = batch['valid_elem'].detach().cpu()
            target = batch['target'].detach().cpu()
            loss = self.get_loss(output, target)
            gen_loss.append(loss.item())
            metrics(valid_elem, output[:, -1, :])
            iterator.set_description('Test phase, Loss: {:.4f}'.format(np.mean(gen_loss)))

        logger.info(f"Time unit is {loader.dataset._time_unit}")
        metrics_dict = metrics.log()
        # wandb.log(metrics_dict)
        # self.save(metrics_dict)

        return metrics_dict

    def save(self, epoch_metric):
        metrics_str = '_'.join([f'{epoch_metric[key]:.2f}' for key in epoch_metric.keys()])
        torch.save(self.state_dict(), f'models/{self.name + metrics_str}')

    def get_dataset(self, *args, **kwargs):
        return LSTMDataset(*args, **kwargs)


class LSTMModel(BaseModel):

    def __init__(self,
                 num_items,
                 num_geo,
                 device='cpu',
                 use_geo=False,
                 hidden_dim: int = 64,
                 max_time_span: int = 64):
        super().__init__(device=device,
                         num_items=num_items,
                         hidden_dim=hidden_dim,
                         max_time_span=max_time_span)

        self._lstm = torch.nn.LSTM(
            hidden_dim, hidden_dim,
            batch_first=True,
            dropout=0.2
        ).to(device)

        self.vec2lstm = torch.nn.Linear(hidden_dim, hidden_dim).to(device)

        self.use_geo = use_geo

    def encode(self, inp):
        encoded_seq = self._item_embedding(inp['items'].long())
        lstm_out = self._lstm(encoded_seq)[0]

        return lstm_out
