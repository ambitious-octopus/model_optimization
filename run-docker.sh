docker build -t model-optimization -f docker/Dockerfile.tutorials docker
docker run -it --net host --gpus all -v $(pwd):/workspace --ipc=host model-optimization bash