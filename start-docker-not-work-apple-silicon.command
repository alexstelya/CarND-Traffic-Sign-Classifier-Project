#!/bin/bash
#Run a Jupyter notebook server with your own notebook directory and mount it /tf/notebooks
docker run -it --rm -p 8888:8888 -v $(pwd):/tf/notebooks tensorflow/tensorflow:latest-jupyter 