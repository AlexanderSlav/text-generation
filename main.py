import argparse
from train import TextGenerationModel
import pytorch_lightning as pl
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--txt', type=str, help='path to txt file', required=True)
    parser.add_argument('-seq_len', '--sequence_length', type=int, default=59, help='length of input sequence')
    parser.add_argument('--layers', default=2, type=int, help='Num of RNN layers')
    parser.add_argument('-hs', '--hidden_size', default=256, type=int, help='hidden size')
    parser.add_argument('-lr', '--initial_lr', type=float, default=0.001, help='set initial learning rate')
    parser.add_argument('-max_epoch', type=int, default=1000, help='max epoch amount')
    parser.add_argument('-w', '--workers', type=int, default=0, help='workers amount')
    parser.add_argument('-batch', '--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    txt_path = args.txt
    seq_len = args.sequence_length
    time_now = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    logger_dir = f'checkpoints/{time_now}'

    model = TextGenerationModel(
        txt=txt_path,
        seq_len=seq_len,
        layers=args.layers,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.initial_lr,
        workers=args.workers
    )

    trainer = pl.Trainer(
        default_root_dir=logger_dir,
        # checkpoint_callback=save_every_n,
        # early_stop_callback=early_stop_callback,
        max_epochs=args.max_epoch, gpus=[0],
        # logger=[logger, all_loggers],
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()