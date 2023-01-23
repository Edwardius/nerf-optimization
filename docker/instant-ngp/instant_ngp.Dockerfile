FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# ================= Dependencies ===================
# python
RUN apt-get update -y
ENV http_proxy $HTTPS_PROXY
ENV https_proxy $HTTPS_PROXY

RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# instant-ngp dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN pip install cmake==3.22

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    libopenexr-dev \
    libxi-dev \
    libxrandr-dev \
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

# ================= User & Environment Setup, Repos ===================
WORKDIR /project
ENV DEBIAN_FRONTEND interactive
ENV PATH="/usr/local/cuda-11.4/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV TCNN_CUDA_ARCHITECTURES 86

RUN apt-get update && apt-get install -y curl sudo && \
    rm -rf /var/lib/apt/lists/*

# Add a docker user so that created files in the docker container are owned by a non-root user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Remap the docker user and group to be the same uid and group as the host user.
# Any created files by the docker container will be owned by the host user.
RUN USER=docker && \
    GROUP=docker && \                                                                     
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \                                                                                                            
    chown root:root /usr/local/bin/fixuid && \                                                                              
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \                                                                                               
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

ENTRYPOINT [ "/usr/local/bin/fixuid"]
CMD ["sleep", "inf"]