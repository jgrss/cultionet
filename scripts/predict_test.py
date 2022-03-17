#!/usr/bin/env python

import argparse

import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths

import numpy as np
from rasterio.windows import Window
import torch


def stack_to_file(out_path: str, stack: np.ndarray) -> None:
    with open(out_path, mode='wb') as f:
        np.save(f, stack)


def predict_test(args):
    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path)

    # Load the z-score norm values
    data_values = torch.load(ppaths.norm_file)

    # Create the train data object
    ds = EdgeDataset(
        ppaths.train_path,
        data_means=data_values.mean,
        data_stds=data_values.std
    )

    ds.shuffle_items()

    # Apply inference
    stack, lit_model = cultionet.predict(
        predict_ds=ds[:1],
        ckpt_file=ppaths.ckpt_file,
        filters=args.filters,
        device=args.device,
        w=Window(row_off=0, col_off=0, height=ds[0].height, width=ds[0].width),
        w_pad=Window(row_off=0, col_off=0, height=ds[0].height, width=ds[0].width)
    )

    boundary_dist = ds[0].bdist.detach().cpu().numpy().reshape(ds[0].height, ds[0].width)
    labels = ds[0].y.detach().cpu().numpy().reshape(ds[0].height, ds[0].width)
    stack = np.vstack((stack, boundary_dist[None], labels[None]))
    stack_to_file(args.out_path, stack)


def main():
    parser = argparse.ArgumentParser(description='Predicts a test object',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="########\n"
                                            "Examples\n"
                                            "########\n\n"
                                            "python predict_test.py --project-path /projects/data -o file.npy \n\n")

    parser.add_argument('-p', '--project-path', dest='project_path', help='The project path', default=None)
    parser.add_argument('-o', '--out-path', dest='out_path', help='The output path', default=None)
    parser.add_argument('--filters', dest='filters', help='The number of base filters', default=32, type=int)
    parser.add_argument('--device', dest='device', help='The device to train on', default='gpu', choices=['cpu', 'gpu'])

    args = parser.parse_args()

    predict_test(args)


if __name__ == '__main__':
    main()
