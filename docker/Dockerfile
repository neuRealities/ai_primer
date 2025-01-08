#Non-CUDA
FROM jupyter/minimal-notebook

#CUDA
#FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

WORKDIR /srv
COPY ./docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt