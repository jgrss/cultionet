FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install -U pip setuptools wheel
RUN pip install -U --no-cache-dir "setuptools<=58.*"

# Install PyTorch Geometric and its dependencies
RUN pip install torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Install GDAL
RUN apt update -y && \
    apt upgrade -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y && \
    apt update -y && \
    apt install \
    build-essential \
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

# RUN pip install -U --no-cache-dir cython>=0.29.*
RUN pip install -U --no-cache-dir numpy>=1.19.0

RUN GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2"."$3}') && \
    pip install GDAL==$GDAL_VERSION --no-binary=gdal

# Install cultionet
RUN pip install -U git+https://github.com/jgrss/geowombat.git@jgrss/docker
RUN pip install -U git+https://github.com/jgrss/cultionet.git@jgrss/docs

CMD ["cultionet"]
