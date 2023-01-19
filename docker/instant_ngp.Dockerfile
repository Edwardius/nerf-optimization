FROM nvidia/cuda:11.4.3-base-ubuntu20.04
WORKDIR /project

# python
RUN apt-get update -y
ENV http_proxy $HTTPS_PROXY
ENV https_proxy $HTTPS_PROXY
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# instant-ngp dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN pip install cmake --upgrade

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    libopenexr-dev \
    libxi-dev \
    libglfw3-dev \
    libglew-dev \
    libomp-dev \
    libxinerama-dev \
    libxcursor-dev --no-install-recommends

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade

RUN pip install \ 
    commentjson \
    imageio \
    numpy \
    opencv-python-headless \
    pybind11 \
    pyquaternion \
    scipy \
    tqdm 

ENV PATH="/usr/local/cuda-11.4/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"