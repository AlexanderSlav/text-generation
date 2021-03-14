
import torch
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
import string
import random
import torch.nn.functional as F
from models.model import CharLSTM
from engine.dataloader import TextDataset
import numpy as np



class TextGenerationModel(pl.LightningModule):
    def __init__(self, txt, seq_len, layers, batch_size, hidden_size, lr, workers):
        super(TextGenerationModel, self).__init__()
        self.save_hyperparameters()
        self.train_dataset = TextDataset(txt_path=txt, seq_len=seq_len)
        self.val_dataset = TextDataset(txt_path=txt, mode='val', seq_len=seq_len)
        self.metric = pl.metrics.Accuracy()
        self.seq_len = seq_len
        tokens = self.train_dataset.chars
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
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worksers,
            pin_memory=True,
            # drop_last=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.val_dataset,
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

    def validation_step(self, batch, *args):
        x, y = batch
        x.squeeze_()
        y.squeeze_()

        y_pred, (self.state_h, self.state_c) = self.forward(x)
        val_loss = self.loss_fn(y_pred, y.flatten())
        # self.train_acc(F.softmax(y_pred.transpose(1, 2), dim=0), y)
        self.state_c = self.state_c.detach()
        self.state_h = self.state_h.detach()
        val_accuracy = self.metric(torch.softmax(y_pred.reshape(self.seq_len, -1, self.val_dataset.n_symbols).permute(1, 2, 0), dim=1), y.permute(1, 0))
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return {'val_loss': val_loss, "val_acc": val_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_metric = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.log("avg_accuracy", avg_metric)
        # logs = {'avg_loss': avg_loss}
        # tensorboard_logs = {'train/avg_loss': avg_loss}
        # results = {'log': tensorboard_logs}
        with torch.no_grad():
            print("\n")
            print(self.inference(generate=500))
            print("\n")
        self.train()
        self.state_h, self.state_c = self.model.init_hidden(self.seq_len)

    @torch.no_grad()
    def inference(self, generate, inp=None):
        self.eval()

        if inp is not None:
            chars = [ch for ch in inp]
        else:
            chars = [random.choice(string.ascii_letters[25:])]

        self.state_h, self.state_c = self.model.init_hidden(1)

        for ch in chars:
            char = self.predict(ch, top_k=10)

        chars.append(char)

        for ii in range(generate):
            char = self.predict(chars[-1], top_k=10)
            chars.append(char)

        return ''.join(chars)

    def predict(self, char, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''

        # if h is None:
        #     h = self.init_hidden(1)

        x = torch.tensor([self.model.char2int[char]])
        x = F.one_hot(x, len(self.model.chars))

        inputs = x.unsqueeze(0).cuda().float()

        # h = tuple([each.data for each in h])
        # self.state_h, self.state_c = self.model.init_hidden(self.seq_len)
        out, (self.state_h, self.state_c) = self.forward(inputs)
        # out, h = self.forward(inputs)

        p = F.softmax(out).data

        p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(self.model.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        return self.model.int2char[char]