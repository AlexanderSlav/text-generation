import argparse 
from dataset import TextDataset, TextGenerationModel
from pytorch_lightning.loggers import TensorBoardLogger
import datetime
import pytorch_lightning as pl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seq_len', '--sequence_length', type=int, help='length of input sequence')
    parser.add_argument('-p', '--data_path', type=str, help='path to dataset')
    parser.add_argument('-b_s', '--batch_size', type=int, help='batch size')
    parser.add_argument('-w_num', '--workers', type=int, help='workers amount')
    parser.add_argument('-max_epoch', type=int, help='max epoch amount')
    parser.add_argument('-initial_lr', type=float, help='set initial learning rate')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    time_now = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    logger_dir = f'checkpoints/{time_now}'
    logger = TensorBoardLogger(logger_dir, name='logs', version='')
    model = TextGenerationModel(args)
    trainer = pl.Trainer(
        default_root_dir=logger_dir,
        max_epochs=args.max_epoch, gpus=[0], logger=[logger],
    )

    trainer.fit(model)
if __name__ == "__main__":
    main()

