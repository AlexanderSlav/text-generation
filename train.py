
import torch
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss

from models.model import CharLSTM
from engine.dataloader import TextDataset




class TextGenerationModel(pl.LightningModule):
    def __init__(self, txt, seq_len, layers, batch_size, hidden_size, lr, workers):
        super(TextGenerationModel, self).__init__()

        self.dataset = TextDataset(txt_path=txt, seq_len=seq_len)
        self.seq_len = seq_len
        tokens = self.dataset.chars
        self.model = CharLSTM(
            tokens=tokens,
            n_layers=layers,
            n_hidden=hidden_size,
        )
        self.num_worksers = workers
        self.batch_size = batch_size
        self.lr = lr
        self.state_h, self.state_c = self.model.init_hidden(self.seq_len)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.model.forward(x, (self.state_h, self.state_c))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return [optimizer] #, [scheduler]

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worksers,
            pin_memory=True,
            # drop_last=True
        )
        return loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.squeeze_()
        y.squeeze_()

        y_pred, (self.state_h, self.state_c) = self.forward(x)
        loss = self.loss_fn(y_pred, y.flatten())
        # self.train_acc(F.softmax(y_pred.transpose(1, 2), dim=0), y)
        self.state_c = self.state_c.detach()
        self.state_h = self.state_h.detach()
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_loss", avg_loss)
        # logs = {'avg_loss': avg_loss}
        # tensorboard_logs = {'train/avg_loss': avg_loss}
        # results = {'log': tensorboard_logs}
        self.state_h, self.state_c = self.model.init_hidden(self.seq_len)
