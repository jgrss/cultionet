FROM nvidia/cuda:11.3.0-base-ubuntu20.04

# Install GDAL
RUN apt update -y && \
    apt upgrade -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y && \
    apt update -y && \
    apt install \
    build-essential \
    python3.8 \
    python3-pip \
    libgeos++-dev \
    libgeos-3.8.0 \
    libgeos-c1v5 \
    libgeos-dev \
    libgeos-doc \
    libspatialindex-dev \
    g++ \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libspatialindex-dev \
    geotiff-bin \
    libgl1 \
    git -y

RUN export CPLUS_INCLUDE_PATH="/usr/include/gdal"
RUN export C_INCLUDE_PATH="/usr/include/gdal"
RUN export LD_LIBRARY_PATH="/usr/local/lib"

RUN pip install -U pip setuptools wheel
RUN pip install -U --no-cache-dir "setuptools<=58.*"
RUN pip install -U --no-cache-dir cython>=0.29.*
RUN pip install -U --no-cache-dir numpy>=1.21.0

# Install PyTorch Geometric and its dependencies
RUN pip install \
    torch \
    torchvision \
    torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN TORCH_VERSION=`(python -c "import torch;print(torch.__version__)")` &&
    pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html

RUN GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2"."$3}') && \
    pip install GDAL==$GDAL_VERSION --no-binary=gdal

# Install cultionet
RUN pip install --user cultionet@git+https://github.com/jgrss/cultionet.git

CMD ["cultionet"]
