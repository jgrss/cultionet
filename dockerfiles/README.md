## Build a Docker image

1. Clone `cultionet`
```commandline
git clone https://github.com/jgrss/cultionet.git
cd cultionet/
```

2. Build the image

In this command, replace <image name> with the name of the new image.
```commandline
docker build -t <image name> .
```

You can also add a tag. For example, 
```commandline
docker build -t cultionet:v0.1 .
```

---
> **NOTE**: Be patient -- the image can take a while to build.
---

3. Run `cultionet` with the new Docker image.
```commandline
docker run -it <image:tag> cultionet [commands]
```

For example, to run the help printout:
```commandline
docker run -it cultionet:v0.1 cultionet -h
```

If you saved the image without a tag then:
```commandline
docker run -it cultionet:latest cultionet -h
```
