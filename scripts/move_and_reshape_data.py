import argparse
from pathlib import Path

import joblib
import torch
from einops import rearrange
from tqdm import tqdm

from cultionet.data.data import Data


def reshape_batch(filename: Path) -> Data:
    # Load the old file
    batch = joblib.load(filename)

    batch_x = rearrange(
        batch.x,
        '(h w) (c t) -> 1 c t h w',
        c=batch.nbands,
        t=batch.ntime,
        h=batch.height,
        w=batch.width,
    )
    batch_y = rearrange(
        batch.y, '(h w) -> 1 h w', h=batch.height, w=batch.width
    )
    batch_bdist = rearrange(
        batch.bdist, '(h w) -> 1 h w', h=batch.height, w=batch.width
    )

    return Data(
        x=batch_x,
        y=batch_y,
        bdist=batch_bdist,
        start_year=torch.tensor([batch.start_year]).long(),
        end_year=torch.tensor([batch.end_year]).long(),
        left=torch.tensor([batch.left]).float(),
        bottom=torch.tensor([batch.bottom]).float(),
        right=torch.tensor([batch.right]).float(),
        top=torch.tensor([batch.top]).float(),
        res=torch.tensor([batch.res]).float(),
        batch_id=[batch.train_id],
    )


def read_and_move(
    input_data_path: str,
    output_data_path: str,
):
    input_data_path = Path(input_data_path)
    output_data_path = Path(output_data_path)
    output_data_path.mkdir(parents=True, exist_ok=True)

    # Get raw data only
    data_list = list(input_data_path.glob("*_none.pt"))

    for fn in tqdm(data_list, desc='Moving files'):
        new_batch = reshape_batch(fn)
        new_batch.to_file(output_data_path / fn.name)


def main():
    parser = argparse.ArgumentParser(
        description="Move and reshape data batches",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-data-path",
        dest="input_data_path",
        help="The input path of data to reshape (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "--output-data-path",
        dest="output_data_path",
        help="The output path of reshaped data (default: %(default)s)",
        default=None,
    )

    args = parser.parse_args()

    read_and_move(
        input_data_path=args.input_data_path,
        output_data_path=args.output_data_path,
    )


if __name__ == '__main__':
    main()
