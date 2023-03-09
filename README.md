[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub version](https://badge.fury.io/gh/jgrss%2Fcultionet.svg)](https://badge.fury.io/gh/jgrss%2Fcultionet)
[![](https://github.com/jgrss/cultionet/actions/workflows/ci.yml/badge.svg)](https://github.com/jgrss/cultionet/actions?query=workflow%3ACI)

**cultionet** is a library for semantic segmentation of cultivated land using a neural network. There are various model configurations that can
be used in `cultionet`, but the base architecture is [UNet 3+](https://arxiv.org/abs/2004.08790) with [multi-stream decoders](https://arxiv.org/abs/1902.04099).

The library is built on **[PyTorch Lightning](https://www.pytorchlightning.ai/)**. The segmentation objectives (class targets and losses) were designed following [previous work](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301115).

Below are highlights of Cultionet:

1. satellite image time series instead of individual dates for training and inference
2. [UNet 3+](https://arxiv.org/abs/2004.08790) [Psi](https://arxiv.org/abs/1902.04099) residual convolution (`ResUNet3Psi`) architecture
3. [Spatial-channel attention](https://www.mdpi.com/2072-4292/14/9/2253)
4. [Tanimoto loss](https://www.mdpi.com/2072-4292/13/18/3707)
5. Deep supervision and temporal features with [RNN STAR](https://www.sciencedirect.com/science/article/pii/S0034425721003230)
6. Deep, multi-output supervision

## The cultionet input data

The model inputs are satellite time series (e.g., bands or spectral indices). Data are stored in a [PyTorch Geometric Data object](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data). For example, `cultionet` datasets will have data
that look something like the following.

```python
from torch_geometric.data import Data

Data(
  x=[10000, 65], y=[10000], bdist=[10000],
  height=100, width=100, ntime=13, nbands=5,
  zero_padding=0, start_year=2020, end_year=2021,
  left=<longitude>, bottom=<latitude>,
  right=<longitude>, top=<latitude>,
  res=10.0, train_id='{site id}_2021_1_none', num_nodes=10000
)
```

where

```
x = input features = torch.Tensor of (samples x bands*time)
y = labels = torch.Tensor of (samples,)
bdist = distance transform = torch.Tensor of (samples,)
height = image height/rows = int
width = image width/columns = int
ntime = image time dimensions/sequence length = int
nbands = image band dimensions/channels = int
left = image left coordinate bounds = float
bottom = image bottom coordinate bounds = float
right = image right coordinate bounds = float
top = image top coordinate bounds = float
res = image spatial resolution = float
train_id = image id = str
```

As an example, for a time series of red, green, blue, and NIR with 25 time steps (bi-weekly + 1 additional end point),
the data would be shaped like:

```
x = [[r_w1, ..., r_w25, g_w1, ..., g_wN, b_w1, ..., b_wN, n_w1, ..., n_wN]]
```

## Create train dataset

### Create the training data

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
> training grid (described above). Thus, it is important to exhaustively digitize all crop fields within a grid.

**Configuration file**
> The configuration file (`cultionet/scripts/config.yml`) is used to create training datasets. This file is only meant
> to be a template. For each project, copy this template and modify it accordingly.

* image_vis
  * A list of image indices to use for training.
* regions
  * The start and end range of the training regions to use in the dataset.
* years
  * A list of years to use in the training dataset. Image years correspond to the _end_ period of the time series.
  Thus, 2021 would align with a time 2020-2021 series.

**Training data requirements**
> The polygon vector file should have a field with values for crop fields set equal to 1. Other crop classes are allowed and
> can be recoded during the data creation step. However, the current version of cultionet expects the final data to be binary
> (i.e., 0=non-cropland; 1=cropland). For grids with all null data (i.e., non-crop), simply create an empty grid file.

**Training name requirements**
> The polygon/grid pairs should be named with the format **{region}_{poly}_{year}.gpkg**. The region name can be any string
> or integer. However, integers should have six character length (e.g., the region might correspond to grid 1 and be
> named '000001_poly_2020.gpkg'.

Example directory structure and format for training data. For a single AOI, there is a grid file and a polygon file. The
number of grid/polygon pairs is unlimited.

```yaml
project_dir:
  user_train:
    '{region}_grid_{time_series_end_year}.gpkg'
    '{region}_poly_{time_series_end_year}.gpkg'
```

Using the format above, a train directory might look like:

```yaml
project_dir:
  user_train:
    'site1_grid_2021.gpkg'
    'site1_poly_2021.gpkg'
    'site1_grid_2022.gpkg'
    'site1_poly_2022.gpkg'
    'site2_grid_2020.gpkg'
    'site2_poly_2020.gpkg'
    ...
```

or

```yaml
project_dir:
  user_train:
    '000001_grid_2021.gpkg'
    '000001_poly_2021.gpkg'
    '000001_grid_2022.gpkg'
    '000001_poly_2022.gpkg'
    '000002_grid_2020.gpkg'
    '000002_poly_2020.gpkg'
    ...
```

> **Note:** a site can have multiple grid/polygon pairs if collected across different timeframes

### Create the image time series

This must be done outside of `cultionet`. Essentially, a directory with band or VI time series must be generated before
using `cultionet`.

> **Note:** it is expected that the time series have length greater than 1

- The raster files should be stored as GeoTiffs with names that follow a date format (e.g., `yyyyddd.tif` or `yyymmdd.tif`).
  - The date format can be specified at the CLI.
- There is no maximum requirement on the temporal frequency (i.e., daily, weekly, bi-weekly, monthly, etc.).
  - Just note that a higher frequency will result in larger memory footprints for the GPU, plus slower training and inference.
- While there is no requirement for the time series frequency, time series _must_ have different start and end years.
  - For example, a northern hemisphere time series might consist of (1 Jan 2020 to 1 Jan 2021) whereas a southern hemisphere time series might range from (1 July 2020 to 1 July 2021). In either case, note that something like (1 Jan 2020 to 1 Dec 2020) will not work.
- The years in the directories must align with the training data files. More specifically, the training data year (year in the polygon/grid pairs) should correspond to the time series end year.
  - For example, a file named `000001_poly_2020.gpkg` should be trained on 2019-2020 imagery, while `000001_poly_2022.gpkg` would match a 2021-2022 time series.
- The image time series footprints (bounding box) can be of any size, but should encompass the training data bounds. During data creation (next step below), only the relevant bounds of the image are extracted and matched with the training data using the training grid bounds.

**Example time series directory with bi-weekly cadence for three VIs (i.e., evi2, gcvi, kndvi)**

```yaml
project_dir:
   time_series_vars:
      region:
         evi2:
          2020001.tif
          2020014.tif
          ...
          2021001.tif
          2021014.tif
          ...
          2022001.tif
         gcvi:
          <repeat of above>
         kndvi:
          <repeat of above>
```

### Create the time series training data

After training data and image time series have been created, the training data PyTorch files (.pt) can be generated using the commands below.

> **Note:** Modify a copy of `cultionet/scripts/config.yml` as needed.

```commandline
# Navigate to the cultionet script directory.
cd cultionet/scripts/
# Activate the virtual environment. See installation section below for environment details.
pyenv venv venv.cultionet
# Create the training dataset.
(venv.cultionet) cultionet create --project-path /project_dir --grid-size 100 100 --config-file config.yml
```

The output .pt data files will be stored in `/project_dir/data/train/processed`. Each .pt data file will consist of
all the information needed to train the segmentation model.

## Training a model

To train the model, you will need to create the train dataset object and pass it to the `cultionet` fit method. A script
is provided to help ease this process. To train a model on a dataset, use (as an example):

```commandline
(venv.cultionet) cultionet train --project-path /project_dir --val-frac 0.2 --random-seed 500 --batch-size 4 --epochs 30 --filters 32 --device gpu --patience 5 --learning-rate 0.001 --reset-model
```

For more CLI options, see:

```commandline
(venv.cultionet) cultionet train -h
```

After a model has been fit, the last checkpoint file can be found at `/project_dir/ckpt/last.ckpt`.

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

### (Option 1) Build Docker images

If using a GPU with CUDA 11.3, see the `cultionet` [Dockerfile](https://github.com/jgrss/cultionet/blob/main/Dockerfile)
and [dockerfiles/README.md](https://github.com/jgrss/cultionet/blob/main/dockerfiles/README.md) to build a Docker image.

If installing from scratch locally, see the instructions below.

### (Option 2) Install with Conda Mamba on a CPU

#### 1) Create a Conda `environment.yml` file with:

```yaml
name: venv.cnet
channels:
- defaults
dependencies:
- python=3.8.12
- libgcc
- libspatialindex
- libgdal=3.4.1
- gdal=3.4.1
- numpy>=1.22.0
- pip
```

#### 2) Install Python packages

```commandline
conda install -c conda-forge mamba
conda config --add channels conda-forge
mamba env create --file environment.yml
conda activate venv.cnet
(venv.cnet) mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
(venv.cnet) mamba install pyg -c pyg
(venv.cnet) pip install -U pip setuptools wheel
(venv.cnet) pip install cultionet@git+https://github.com/jgrss/cultionet.git
```

### (Option 3) Install with pip on a CPU

This section assumes you have all the necessary Linux builds, such as GDAL. If not, see the next installation section.

#### Install Python packages

```commandline
pyenv virtualenv 3.8.12 venv.cnet
pyenv activate venv.cnet
(venv.cnet) pip install -U pip setuptools wheel numpy cython
(venv.cnet) pip install gdal==$(gdal-config --version | awk -F'[.]' '{print $1"."$2"."$3}') --no-binary=gdal
(venv.cnet) pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
(venv.cnet) TORCH_VERSION=$(python -c "import torch;print(torch.__version__)")
(venv.cnet) pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
(venv.cnet) pip install cultionet@git+https://github.com/jgrss/cultionet.git
```

### (Option 4) Install CUDA and built GPU packages

1. Install NVIDIA driver (skip if using the CPU)

```commandline
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt install nvidia-driver-465
```

`reboot machine`

2. Install CUDA toolkit (skip if using the CPU)
> See https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

`reboot machine`

3. Install Pyenv
> See https://github.com/pyenv/pyenv/wiki#suggested-build-environment
```commandline
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

4. Add to the .bashrc:
```commandline
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if which pyenv > /dev/null; then eval "$(pyenv init --path)"; fi
if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
if which pyenv > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi
```

5. Install new version of Python
```commandline
pyenv install 3.8.12
```

6. Create a new virtual environment
```commandline
pyenv virtualenv 3.8.12 venv.cultionet
```

7. Install libraries
```commandline
pyenv activate venv.seg
```

8. Update install libraries
```commandline
(venv.cultionet) pip install -U pip setuptools wheel "cython>=0.29.*" "numpy<=1.21.0"
# required to build GDAL Python bindings for 3.2.1
(venv.cultionet) pip install --upgrade --no-cache-dir "setuptools<=58.*"
```

9. Install PyTorch
> See https://pytorch.org/get-started/locally/
```commandline
(venv.cultionet) pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```commandline
python -c "import torch;print(torch.cuda.is_available())"
```

10. Install PyTorch geometric dependencies
```commandline
(venv.cultionet) pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-geometric-temporal -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
```

11. Install GDAL
```commandline
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt install build-essential
sudo apt update
sudo apt install libspatialindex-dev libgdal-dev gdal-bin

export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

12. Install GDAL Python bindings
```commandline
(venv.cultionet) pip install GDAL==3.2.1
```

### Package

Install `cultionet`

```commandline
git clone git@github.com:jgrss/cultionet.git
cd cultionet
(venv.cultionet) pip install .
```
