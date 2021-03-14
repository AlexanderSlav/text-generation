import argparse
from train import TextGenerationModel
import pytorch_lightning as pl
import datetime
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--txt', type=str, help='path to txt file', required=True)
    parser.add_argument('-seq_len', '--sequence_length', type=int, default=60, help='length of input sequence')
    parser.add_argument('--layers', default=2, type=int, help='Num of RNN layers')
    parser.add_argument('-hs', '--hidden_size', default=256, type=int, help='hidden size')
    parser.add_argument('-lr', '--initial_lr', type=float, default=0.001, help='set initial learning rate')
    parser.add_argument('-max_epoch', type=int, default=50, help='max epoch amount')
    parser.add_argument('-w', '--workers', type=int, default=4, help='workers amount')
    parser.add_argument('-batch', '--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('-test', action='store_true', help='inference mode on')
    parser.add_argument('-chk', '--checkpoint', help='path to checkpoint (for inference only)')
    parser.add_argument('--use_gru', action='store_true', help='use lstm instead of gru')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    txt_path = args.txt
    seq_len = args.sequence_length
    time_now = datetime.datetime.now().strftime("%H:%M")
    logger_dir = f'checkpoints/{time_now}_seq_len_{args.sequence_length}_n_layers_{args.layers}_lr_{args.initial_lr}_hidden_state_{args.hidden_size}'

    model = TextGenerationModel(
        txt=txt_path,
        seq_len=seq_len,
        layers=args.layers,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.initial_lr,
        workers=args.workers,
        use_gru=args.use_gru,
    )

    if not args.test:
        trainer = pl.Trainer(
            default_root_dir=logger_dir,
            # checkpoint_callback=save_every_n,
            # early_stop_callback=early_stop_callback,
            max_epochs=args.max_epoch, gpus=[0],
            # logger=[logger, all_loggers],
        )

        trainer.fit(model)

    if args.checkpoint is not None:
        # checkpoint = torch.load(args.checkpoint)['state_dict']
        model = TextGenerationModel.load_from_checkpoint(args.checkpoint).cuda()
        # model.load_state_dict(checkpoint)

    result = model.inference(1000, "The")
    print(result)

if __name__ == '__main__':
    main()