FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
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

# kilonerf dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    libgl-dev \
    freeglut3-dev --no-install-recommends

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade

RUN pip install \ 
    numpy \
    scikit-image \
    scipy \
    tqdm \
    imageio \
    pyyaml \
    imageio-ffmpeg \
    lpips \
    opencv-python 

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

COPY src/kilonerf/cuda/dist/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl \
     /home/docker/kilonerf-cuda-ext/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl
RUN pip install /project/kilonerf-cuda-ext/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl

# ================= User & Environment Setup, Repos ===================
ENV DEBIAN_FRONTEND int`/home/e23zhou/code/test`eractive
ENV PATH="/usr/local/cuda-11.1/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV TCNN_CUDA_ARCHITECTURES 86
ENV KILONERF_HOME "/project/kilonerf"

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
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/\n  - /opt/venv/" > /etc/fixuid/config.yml

USER docker:docker
WORKDIR /home/docker/

ENTRYPOINT ["/usr/local/bin/fixuid"]
CMD ["sleep", "inf"]