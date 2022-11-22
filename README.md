[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub version](https://badge.fury.io/gh/jgrss%2Fcultionet.svg)](https://badge.fury.io/gh/jgrss%2Fcultionet)
[![](https://github.com/jgrss/cultionet/actions/workflows/ci.yml/badge.svg)](https://github.com/jgrss/cultionet/actions?query=workflow%3ACI)

**cultionet** is a library for semantic segmentation of cultivated land using a neural network.

The library is built on **[PyTorch Lightning](https://www.pytorchlightning.ai/)**. The segmentation objectives (class targets and losses) were designed following the work by [Waldner _et al._](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301115). However, this library differs from the paper above:

1. **cultionet** uses time series instead of individual dates for training and inference.
2. **cultionet** uses a different CNN architecture.

## The cultionet input data

The model inputs are satellite time series--they can be either bands or spectral indices. Data are stored in a [PyTorch Geometric Data object](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data). For example,

```python
from torch_geometric.data import Data

batch = Data(x=x, y=y, edge_index=edge_index, edge_attrs=edge_attrs)
```

where

```
x = torch.Tensor of (samples x bands*time)
y = torch.Tensor of (samples,)
```

As an example, for a time series of red, green, blue, and NIR with 25 time steps (bi-weekly + 1 additional end point), the data would be shaped like:

```
x = [[r_w1, ..., r_w25, g_w1, ..., g_wN, b_w1, ..., b_wN, n_w1, ..., n_wN]]
```

## Create train dataset

### Create the training data

Training data should consist of two files per grid/year. One file is a polygon vector file (stored as a GeoPandas-compatible
format like GeoPackage) of the training grid for a region. The other is a polygon vector file (stored in the same format)
of the training labels for a grid.

**What is a grid?**
> A grid defines an area to be labeled. For example, a grid could be 1 km x 1 km. A grid should be small enough to be combined
> with other grids in batches in GPU memory. Thus, 1 km x 1 km is a good size with, say, Sentinel-2 imagery at 10 m spatial
> resolution.

> **Note:** grids across a study area should all be of equal dimensions

**What is a training label?**
> Training labels are __polygons__ of delineated crop (i.e., crop fields). The training labels will be clipped to the
> training grid (described above). Thus, it is important to exhaustively digitize all crop fields within a grid.

**Configuration file**
> The configuration file (`cultionet/scripts/config.yml`) is used to create training datasets. This file is only meant
> to be a template. For each project, copy this template and modify it accordingly.

* image_vis
  * A list of image indices to use for training.
* regions
  * The start and end range of the training regions to use in the dataset.
* years
  * A list of years to use in the training dataset. Image years correspond to the _end_ period of the time series. Thus, 2021 would align with a time 2020-2021 series.

**Training data requirements**
> The polygon vector file should have a field named 'class', with values for crop fields set equal to 1. For grids with
> all null data (i.e., non-crop), simply create an empty polygon or a polygon matching the grid extent with 'class'
> value equal to 0.

**Training name requirements**
> The polygon/grid pairs should be named **{region}_{poly}_{year}.gpkg**. The region name should be an integer of
> six character length (e.g., the region might correspond to grid 1 and be named '000001_poly_2020.gpkg'.

Example directory structure for training data.

```yaml
project_dir:
  user_train:
    '{region}_grid_{time_series_end_year}.gpkg'
    '{region}_poly_{time_series_end_year}.gpkg'
```

### Create the image time series

This must be done outside of `cultionet`. Essentially, a directory with band or VI time series must be generated before
using `cultionet`.

> **Note:** it is expected that the time series have length greater than 1

- The raster files should be stored as GeoTiffs with names that follow the `yyyyddd.tif` format.
- There is no maximum requirement on the temporal frequency (i.e., daily, weekly, bi-weekly, monthly, etc.).
  - Just note that a higher frequency will result in larger memory footprints for the GPU plus slower training and inference.
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

To train the model, you will need to create the train dataset object and pass it to `cultionet` fit method. A script
is provided to help ease this process. To train a model on a dataset, use (as an example):

```commandline
(venv.cultionet) cultionet train --project-path /project_dir --val-frac 0.2 --random-seed 500 --batch-size 4 --epochs 30 --filters 32 --device gpu --patience 5 --learning-rate 0.001 --reset-model
```

For more CLI options, see:

```commandline
(venv.cultionet) cultionet train -h
```

### Example usage of the cultionet API

In the examples below, we use the project path of the setup examples above to train a model using cultionet. Note that
this is similar to using the CLI example above. The point here is simply to demonstrate the use of the Python API.

#### Fit a model using cultionet

The example below illustrates what `cultionet train` does.

```python
import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
from cultionet.utils.normalize import get_norm_values

# Fraction of data to use for model validation
# The remainder will be used for training
val_frac = 0.2
# The random seed|state used to split the data
random_seed = 42

# This is a helper function to manage paths
ppaths = setup_paths('project_dir')

# Get the normalization means and std. deviations
ds = EdgeDataset(ppaths.train_path)
cultionet.model.seed_everything(random_seed)
train_ds, val_ds = ds.split_train_val(val_frac=val_frac)
# Calculate the values needed to transform to z-scores, using
# the training data
data_values = get_norm_values(dataset=train_ds, batch_size=16)

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
   val_frac=val_frac,
   epochs=30,
   learning_rate=0.001,
   filters=32,
   random_seed=random_seed
)
```

After a model has been fit, the last checkpoint file can be found at `/project_dir/ckpt/last.ckpt`.

## Predicting on an image with a trained model

```commandline
(venv.cultionet) cultionet predict --project-path /project_dir --out-path predictions.tif --grid-id 1 --window-size 100 --config-file project_config.yml --device cpu --processes 4
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
