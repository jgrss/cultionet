[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![](https://img.shields.io/github/v/release/jgrss/cultionet?display_name=release)](https://github.com/jgrss/cultionet/releases)
[![](https://github.com/jgrss/cultionet/actions/workflows/ci.yml/badge.svg)](https://github.com/jgrss/cultionet/actions?query=workflow%3ACI)

## Cultionet

Cultionet is a library for semantic segmentation of cultivated land with a neural network. The base architecture is a UNet variant, inspired by [UNet 3+](https://arxiv.org/abs/2004.08790) and [Psi-Net](https://arxiv.org/abs/1902.04099), with convolution blocks following [ResUNet-a](https://arxiv.org/abs/1904.00592). The library is built on [PyTorch Lightning](https://www.pytorchlightning.ai/) and the segmentation objectives (class targets and losses) were designed following [previous work in the remote sensing community](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301115).

Key features of Cultionet:

* uses satellite image time series instead of individual dates for training and inference
* uses a [Transformer](https://arxiv.org/abs/1706.03762) time series embeddings
* uses a UNet architecture with skip connections and deep supervision similar to [UNet 3+](https://arxiv.org/abs/2004.08790)
* uses multi-stream outputs inspired by [Psi-Net](https://arxiv.org/abs/1902.04099)
* uses residual [ResUNet-a](https://arxiv.org/abs/1904.00592) blocks with [Dilated Neighborhood Attention](https://arxiv.org/abs/2209.15001)
* uses the [Tanimoto loss](https://www.mdpi.com/2072-4292/13/18/3707)

## Install Cultionet

If PyTorch is installed

```commandline
pip install git@github.com:jgrss/cultionet.git
```

See the [installation section](#installation) for more detailed instructions.

---

## Data format

The model inputs are satellite time series (e.g., bands or spectral indices). Data are stored in a PyTorch [Data](https://github.com/jgrss/cultionet/blob/99fb16797f2d84b812c47dd9d03aea92b6b7aefa/src/cultionet/data/data.py#L51) object. For example, Cultionet datasets will have data that look something like the following.

```python
Data(
  x=[1, 3, 12, 100, 100], y=[1, 100, 100], bdist=[1, 100, 100],
  start_year=torch.tensor([2020]), end_year=torch.tensor([2021]),
  left=torch.tensor([<longitude>]), bottom=torch.tensor([<latitude>]),
  right=torch.tensor([<longitude>]), top=torch.tensor([<latitude>]),
  res=torch.tensor([10.0]), batch_id=['{site id}_2021_1_none'],
)
```

where

```
x = input features = torch.Tensor of (batch x channels/bands x time x height x width)
y = labels = torch.Tensor of (batch x height x width)
bdist = distance transform = torch.Tensor of (batch x height x width)
left = image left coordinate bounds = torch.Tensor
bottom = image bottom coordinate bounds = torch.Tensor
right = image right coordinate bounds = torch.Tensor
top = image top coordinate bounds = torch.Tensor
res = image spatial resolution = torch.Tensor
batch_id = image id = list
```

## Datasets

### Create the vector training dataset

Training data pairs should consist of two files per grid/year. One file is a polygon vector file (stored as a GeoPandas-compatible
format like GeoPackage) of the training grid for a region. The other file is a polygon vector file (stored in the same format)
of the training labels for a grid.

**What is a grid?**
> A grid defines an area to be labeled. For example, a grid could be 1 km x 1 km. A grid should be small enough to be combined
> with other grids in batches in GPU memory. Thus, 1 km x 1 km is a good size with, say, Sentinel-2 imagery at 10 m spatial
> resolution.

> **Note:** grids across a study area should all be of equal dimensions

**What is a training label?**
> Training labels are __polygons__ of delineated cropland (i.e., crop fields). The training labels will be clipped to the
> training grid (described above). Thus, it is important to digitize all crop fields within a grid unless data are to be used
> for partial labels.

**Configuration file**
> The configuration file is used to create training datasets. Copy the [config template](scripts/config.yml) and modify it accordingly.

**Training data requirements**
> The polygon vector file should have a field with values for crop fields set equal to 1. Other crop classes are allowed and
> can be recoded during the data creation step. However, the current version of cultionet expects the final data to be binary
> (i.e., 0=non-cropland; 1=cropland). For grids with all null data (i.e., non-crop), simply create a grid file with no intersecting
> crop polygons.

**Training name requirements**
> There are no requirements. Simply specify the paths in the configuration file.

Example directory structure and format for training data. For each region, there is a grid file and a polygon file. The
number of grid/polygon pairs within the region is unlimited.

```yaml
region_id_file:
  - /user_data/training/grid_REGION_A_YEAR.gpkg
  - /user_data/training/grid_REGION_B_YEAR.gpkg
  - ...

polygon_file:
  - /user_data/training/crop_polygons_REGION_A_YEAR.gpkg
  - /user_data/training/crop_polygons_REGION_B_YEAR.gpkg
  - ...
```

The grid file should contain polygons of the AOIs. The AOIs represent the area that imagery will be clipped and masked to (only 1 km x 1 km has been tested). Required
columns include 'geo_id' and 'year', which are a unique identifier and the sampling year, respectively.

```python
grid_df = gpd.read_file("/user_data/training/grid_REGION_A_YEAR.gpkg")
grid_df.head(2)

 	                                     geo_id year 	    geometry
0 	REGION_A_e3a4f2346f50984d87190249a5def1d0 2021 POLYGON ((...
1 	REGION_A_18485a3271482f2f8a10bb16ae59be74 2021 POLYGON ((...
```

The polygon file should contain polygons of field boundaries, with a column for the crop class. Any number of other columns can be included. Note that polygons do not need to be clipped to the grids.

```python
import geopandas as gpd
poly_df = gpd.read_file("/user_data/training/crop_polygons_REGION_A_YEAR.gpkg")
poly_df.head(2)
 	crop_class      geometry
0          1 POLYGON ((...
1          1 POLYGON ((...
```

### Create the image time series

This must be done outside of Cultionet. Essentially, a directory with band or VI time series must be generated before
using Cultionet.

- The raster files should be stored as GeoTiffs with names that follow a date format (e.g., `yyyyddd.tif` or `yyymmdd.tif`).
  - The date format can be specified at the CLI.
- There is no maximum requirement on the temporal frequency (i.e., daily, weekly, bi-weekly, monthly, etc.).
  - Just note that a higher frequency will result in larger memory footprints for the GPU, plus slower training and inference.
- While there is no requirement for the time series frequency, time series _must_ have different start and end years.
  - For example, a northern hemisphere time series might consist of (1 Jan 2020 to 1 Jan 2021) whereas a southern hemisphere time series might range from (1 July 2020 to 1 July 2021). In either case, note that something like (1 Jan 2020 to 1 Dec 2020) will not work.
- Time series should align with the training data files. More specifically, the training data year (year in the grid vector file) should correspond to the time series start year.
  - For example, a training grid 'year' column equal to 2022 should be trained on a 2022-2023 image time series.
- The image time series footprints (bounding box) can be of any size, but should encompass the training data bounds. During data creation (next step below), only the relevant bounds of the image are extracted and matched with the training data using the training grid bounds.

Example time series directory with bi-weekly cadence for three VIs (i.e., evi2, gcvi, kndvi)

```yaml
project_dir:
   time_series_vars:
      grid_id_a:
         evi2:
          2022001.tif
          2022014.tif
          ...
          2023001.tif
         gcvi:
          <repeat of above>
         kndvi:
          <repeat of above>
      grid_id_b:
        <repeat of above>
```

### Create the time series training dataset

After training data and image time series have been created, the training data PyTorch files (.pt) can be generated using the commands below.

> **Note:** Modify a copy of the [config template](scripts/config.yml) as needed and save in the project directory. The command below assumes image time series are saved under `/project_dir/time_series_vars`. The training polygon and grid paths are taken from the config.yml file.

This command would generate .pt files with image time series of 100 x 100 height/width and a spatial resolution of 10 meters.

```commandline
# Activate your virtual environment. See installation section below for environment details.
pyenv venv venv.cultionet
# Create the training dataset.
(venv.cultionet) cultionet create --project-path /project_dir --grid-size 100 100 --destination train -r 10.0 --max-crop-class 1 --crop-column crop_class --image-date-format %Y%m%d --num-workers 8 --config-file config.yml
```

The output .pt data files will be stored in `/project_dir/data/train/processed`. Each .pt data file will consist of
all the information needed to train the segmentation model.

## Training a model

To train a model on a dataset, use (as an example):

```commandline
(venv.cultionet) cultionet train --val-frac 0.2 --augment-prob 0.5 --epochs 100 --hidden-channels 32 --processes 8 --load-batch-workers 8 --batch-size 4 --accumulate-grad-batches 4 --dropout 0.2 --deep-sup --dilations 1 2 --pool-by-max --learning-rate 0.01 --weight-decay 1e-4 --attention-weights natten
```

For more CLI options, see:

```commandline
(venv.cultionet) cultionet train -h
```

After a model has been fit, the best/last checkpoint file can be found at `/project_dir/ckpt/last.ckpt`.

## Predicting on an image with a trained model

### First, a prediction dataset is needed

```commandline
(venv.cultionet) cultionet create-predict --project-path /project_dir --year 2022 --ts-path /features
--num-workers 4 --config-file project_config.yml
```

### Apply inference over the predictin dataset

```commandline
(venv.cultionet) cultionet predict --project-path /project_dir --out-path predictions.tif --grid-id 1 --window-size 100 --config-file project_config.yml --device gpu --processes 4
```

## Installation

#### Install Cultionet (assumes a working CUDA installation)

1. Create a new virtual environment (example using [pyenv](https://github.com/pyenv/pyenv))
```commandline
pyenv virtualenv 3.10.14 venv.cultionet
pyenv activate venv.cultionet
```

2. Update install numpy and Python GDAL (assumes GDAL binaries are already installed)
```commandline
(venv.cultionet) pip install -U pip
(venv.cultionet) pip install -U setuptools wheel
pip install -U numpy==1.24.4
(venv.cultionet) pip install setuptools==57.5.0
(venv.cultionet) GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2"."$3}')
(venv.cultionet) pip install GDAL==$GDAL_VERSION --no-binary=gdal
```

3. Install PyTorch 2.2.1 for CUDA 11.4 and 11.8
```commandline
(venv.cultionet) pip install -U --no-cache-dir setuptools>=65.5.1
(venv.cultionet) pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

The command below should print `True` if PyTorch can access a GPU.

```commandline
python -c "import torch;print(torch.cuda.is_available())"
```

4. Install `natten` for CUDA 11.8 if using [neighborhood attention](https://github.com/SHI-Labs/NATTEN).
```commandline
(venv.cultionet) pip install natten==0.17.1+torch220cu118 -f https://shi-labs.com/natten/wheels
```

5. Install cultionet

```commandline
(venv.cultionet) pip install git@github.com:jgrss/cultionet.git
```

### Installing CUDA on Ubuntu

See [CUDA installation](docs/cuda_installation.md)
