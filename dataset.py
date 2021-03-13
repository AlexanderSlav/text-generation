import pytorch_lightning as pl
import torch
import pandas as pd
import nltk
import string
from collections import Counter
from models import LSTMBasedModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args =args
        self.dataset = pd.read_csv(self.args.data_path)
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()

        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]


    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
    
    def get_unique_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    
    def load_words(self):
        text = self.dataset.summary.str.cat(sep=' ')
        return text.replace('\n','').split(' ')
    

    def __getitem__(self, index):
        return (
            torch.LongTensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.LongTensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )


class TextGenerationModel(pl.LightningModule):
    
    def __init__(self, args):
        super(TextGenerationModel, self).__init__()
        self.args = args
        self.text_dataset = TextDataset(self.args)
        self.model = LSTMBasedModel(self.text_dataset)
        self.state_h, self.state_c = self.model.init_hidden(self.args.sequence_length)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model.forward(x, (self.state_h, self.state_c))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.initial_lr)
        # scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return [optimizer] #, [scheduler]

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.text_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True
        )
        return loader
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, (self.state_h, self.state_c) = self.forward(x)
        loss = self.loss_fn(y_pred.transpose(1, 2), y)
        self.state_c = self.state_c.detach()
        self.state_h = self.state_h.detach()
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'avg_loss': avg_loss}
        tensorboard_logs = {'train/avg_loss': avg_loss}
        results = {'log': tensorboard_logs}
        self.state_h, self.state_c = self.model.init_hidden(self.args.sequence_length)
        return results


    def val_dataloader(self):
        # SAME
        pass

    def test_dataloader(self):
        # SAME
        pass