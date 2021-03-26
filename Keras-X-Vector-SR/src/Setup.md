Setup
---

For the Kaldi experiment, it is recommended to use the official Kaldi Docker image. Installation is simple and can be accomplished as follows:

1. Download Docker from the official [website](https://www.docker.com/products/docker-desktop).
   1. Install using the default options.
   2. Set the number of cores and memory available to the VM if desired and according to your machine. (The defaults are sensible and can be left alone if desired.)
2. Create a directory for the Kaldi project on your local filesystem. Navigate to this directory in a terminal.
3. Pull the Kaldi Docker image and start it using the following command:
```sh
docker run -v "$PWD":"/home" --rm -it kaldiasr/kaldi:latest
```
> The optional `-v` parameter maps the paths in the subsquent argument from the local filesystem to the guest filesystem using Docker's bind mounts. In this case `$PWD` resolves to the current directory and will be mirrored in `/home` in the active container. This allows data permanence between the container and the host for data sharing. Docker containers are volatile and any data will otherwise be removed on container stopping by the `--rm` argument. See the [documentation](https://docs.docker.com/) for more information.

The Jupyter notebook will run a typical Jupyter/Python environment with the following dependencies:

- TensorFlow >= 2.3
- ffmpg
- matplotlib