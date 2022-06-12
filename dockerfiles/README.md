## Build a Docker image with CUDA 11.3

1. Clone `cultionet`
```commandline
git clone https://github.com/jgrss/cultionet.git
cd cultionet/
```

2. Build the image

```commandline
docker build -t cultionet .
```

---
> **NOTE**: Be patient -- the image can take a while to build.
---

3. Run `cultionet` with the new Docker image.
```commandline
docker run -it cultionet:latest cultionet [commands]
```

For example, to run the help printout:
```commandline
docker run -it cultionet:latest cultionet -h
```

4. Run with GPU
```commandline
docker run -it --rm --gpus=all --runtime=nvidia cultionet:latest cultionet -h
```
