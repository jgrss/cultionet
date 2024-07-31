## (Option 1) Build Docker images

If using a GPU with CUDA 11.3, see the cultionet [Dockerfile](https://github.com/jgrss/cultionet/blob/main/Dockerfile)
and [dockerfiles/README.md](https://github.com/jgrss/cultionet/blob/main/dockerfiles/README.md) to build a Docker image.

If installing from scratch locally, see the instructions below.

## (Option 2) Install locally with GPU

### Install CUDA driver, if necessary

1. Install NVIDIA driver

```commandline
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt install nvidia-driver-465
```

`reboot machine`

2. Install CUDA toolkit
> See https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

`reboot machine`