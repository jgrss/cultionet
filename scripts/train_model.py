#!/usr/bin/env python

import argparse

import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
from cultionet.utils.normalize import get_norm_values
import torch


def train_model(args):
    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path)

    # Check dimensions
    ds = EdgeDataset(ppaths.train_path)
    ds.check_dims()
    # Get the normalization means and std. deviations on the train data
    cultionet.model.seed_everything(args.random_seed)
    train_ds, val_ds = ds.split_train_val(val_frac=args.val_frac)
    # Calculate the values needed to transform to z-scores, using
    # the training data
    data_values = get_norm_values(dataset=train_ds, batch_size=args.batch_size*4)
    torch.save(data_values, str(ppaths.norm_file))

    # Create the train data object again, this time passing
    # the means and standard deviation tensors
    ds = EdgeDataset(
        ppaths.train_path,
        data_means=data_values.mean,
        data_stds=data_values.std
    )

    # Fit the model
    cultionet.fit(
        dataset=ds,
        ckpt_file=ppaths.ckpt_file,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        filters=args.filters,
        random_seed=args.random_seed,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        device=args.device,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.patience
    )


def main():
    parser = argparse.ArgumentParser(description='Trains a model',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="########\n"
                                            "Examples\n"
                                            "########\n\n"
                                            "python train_model.py --project-path /projects/data \n\n")

    parser.add_argument('-p', '--project-path', dest='project_path', help='The project path', default=None)
    parser.add_argument(
        '--val-frac', dest='val_frac', help='The validation fraction (default: %(default)s)', default=0.2, type=float
    )
    parser.add_argument(
        '--random-seed', dest='random_seed', help='The random seed (default: %(default)s)', default=42, type=int
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', help='The batch size (default: %(default)s)', default=4, type=int
    )
    parser.add_argument(
        '--epochs', dest='epochs', help='The number of training epochs (default: %(default)s)', default=30, type=int
    )
    parser.add_argument(
        '--learning-rate', dest='learning_rate', help='The learning rate (default: %(default)s)',
        default=0.001, type=float
    )
    parser.add_argument(
        '--filters', dest='filters', help='The number of base filters (default: %(default)s)', default=32, type=int
    )
    parser.add_argument(
        '--reset-model', dest='reset_model', help='Whether to reset the model (default: %(default)s)',
        action='store_true'
    )
    parser.add_argument(
        '--lr-find', dest='auto_lr_find', help='Whether to tune the learning rate (default: %(default)s)',
        action='store_true'
    )
    parser.add_argument(
        '--device', dest='device', help='The device to train on (default: %(default)s)',
        default='gpu', choices=['cpu', 'gpu']
    )
    parser.add_argument(
        '--gradient-clip-val', dest='gradient_clip_val', help='The gradient clip value (default: %(default)s)',
        default=0.1, type=float
    )
    parser.add_argument(
        '--patience', dest='patience', help='The eartly stopping patience (default: %(default)s)',
        default=7, type=int
    )

    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()

