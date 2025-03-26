#
# Example Dockerfile to build a container image without NGC DLFW image,
#

#
# Requires CUDA 11.0 or later

FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04
ENV BASE_IMAGE_VERSION=nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04

# NCCL: configures to dump debug log as the default
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf


#
# base
#
SHELL ["/bin/bash", "-cu"]

ARG OS_VER=ubuntu18.04
ARG PLATFORM=x86_64

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        openssh-client \
        libcap2 \
        libnuma-dev \
        git \
        curl \
        wget


#
# MOFED 5.1 is required for multi-node training.
#
ARG MOFED_VER=5.1-2.6.2.0
ENV MOFED_VERSION=${MOFED_VER}
ARG MOFED_ARCHIVE=MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}

RUN mkdir -p /tmp/mofed \
    && cd /tmp/mofed \
    && wget -nv http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VER}/${MOFED_ARCHIVE}.tgz \
    && tar -xzvf ${MOFED_ARCHIVE}.tgz --no-same-owner \
    && ${MOFED_ARCHIVE}/mlnxofedinstall --user-space-only --without-fw-update --all -q --force \
    && cd /tmp \
    && rm -rf /tmp/mofed \
    && rm -rf /tmp/MLNX_OFED_LINUX*.logs

RUN dpkg --purge openmpi

#
# OpenMPI
#
ARG OPENMPI_VERSION_BASE=3.1
ENV OPENMPI_VERSION=${OPENMPI_VERSION_BASE}.2
ARG OPENMPI_ARCHIVE=openmpi-${OPENMPI_VERSION}
ARG OPENMPI_PATH=/opt/openmpi

RUN mkdir -p /tmp/openmpi \
    && cd /tmp/openmpi \
    && wget -nv https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_BASE}/${OPENMPI_ARCHIVE}.tar.bz2 \
    && tar xvf ${OPENMPI_ARCHIVE}.tar.bz2 \
    && cd ${OPENMPI_ARCHIVE} \
    && ./configure \
        --prefix=${OPENMPI_PATH} \
        --enable-orterun-prefix-by-default \
        --with-verbs \
        --enable-mpi-thread-multiple \
        CC=gcc \
        CXX=g++ \
    && make -j $(nproc) all \
    && make install \
    && ldconfig \
    && cd /tmp \
    && rm -rf /tmp/openmpi

ENV PATH=${OPENMPI_PATH}/bin:$PATH \
    LD_LIBRARY_PATH=${OPENMPI_PATH}/lib:$LD_LIBRARY_PATH


RUN apt update

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt update && \
    apt install -y python3.8 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --config python
RUN apt install -y python3.8-dev
RUN apt install -y python3-pip
RUN pip install --upgrade pip wheel setuptools requests
RUN python -m pip install --upgrade pip
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install seaborn ipdb  && \
    pip install cmake && \
    HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod && \
    pip install tqdm && \
    pip install ipdb && \
    apt install -y libgl1-mesa-dev

#uninstall+install numpy is a hack for this version [https://github.com/pytorch/pytorch/issues/37377#issuecomment-842038053]  
RUN pip uninstall numpy -y
RUN pip install numpy
# for mmcv compativility
RUN pip uninstall pyparsing -y
RUN pip install pyparsing==2.4.7
RUN pip install piq
RUN pip install optuna
RUN pip install einops
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.0.1"
RUN mim install mmpretrain


RUN pip install transformers  && \
    pip install timm

    
# Bug in huggingface_hub==0.26 and diffusers
RUN pip install huggingface_hub==0.25.0 && \
    pip install diffusers==0.27.2