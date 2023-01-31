FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
ENV CMAKE_VERSION=3.21.0

RUN echo "Installing apt packages..." \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git \
	wget \
	ffmpeg \
	tk-dev \
	libxi-dev \
	libc6-dev \
	libbz2-dev \
	libffi-dev \
	libomp-dev \
	libssl-dev \
	zlib1g-dev \
	libcgal-dev \
	libgdbm-dev \
	libglew-dev \
	qtbase5-dev \
	checkinstall \
	libglfw3-dev \
	libeigen3-dev \
	libgflags-dev \
	libxrandr-dev \
	libopenexr-dev \
	libsqlite3-dev \
	libxcursor-dev \
	build-essential \
	libcgal-qt5-dev \
	libxinerama-dev \
	libboost-all-dev \
	libfreeimage-dev \
	libncursesw5-dev \
	libatlas-base-dev \
	libqt5opengl5-dev \
	libgoogle-glog-dev \
	libsuitesparse-dev \
	python3-setuptools \
	libreadline-gplv2-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# ================= Dependencies ===================
# python
RUN apt-get update -y
ENV http_proxy $HTTPS_PROXY
ENV https_proxy $HTTPS_PROXY

RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    wget \
    libgl-dev \
    build-essential \
    gfortran \
    freeglut3-dev --no-install-recommends

RUN apt-get update -y && apt-get install -y gfortran libopenblas-dev
RUN wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz
RUN tar -zxvf magma-2.5.4.tar.gz

WORKDIR /magma-2.5.4

RUN cp make.inc-examples/make.inc.openblas make.inc
ENV GPU_TARGET "Maxwell Pascal Volta Turing Ampere"
ENV CUDADIR /usr/local/cuda
ENV OPENBLASDIR "/usr"
RUN make
RUN make install prefix=/usr/local/magma

# ================= User, Perms, & Environment Setup ===================
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

USER docker:docker
WORKDIR /home/docker/

# ================= More Kilonerf Dependencies ===================
ENV VIRTUAL_ENV=/home/docker/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade

RUN pip install \ 
    cmake==${CMAKE_VERSION} \
    matplotlib \
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

ENV PATH="/home/docker/:/usr/local/cuda-11.1/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# ================= User, Perms, & Environment Setup ===================
ENV DEBIAN_FRONTEND interactive

ENTRYPOINT ["/usr/local/bin/fixuid", "-q"]

# ================= end command to keep container running ===================
CMD ["sleep", "inf"]