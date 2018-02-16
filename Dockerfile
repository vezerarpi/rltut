FROM chainer/chainer:v4.0.0b3-python3

RUN apt-get update && apt-get install -y \
    cmake                                \
    ffmpeg                               \
    libav-tools                          \
    libboost-all-dev                     \
    libffi-dev                           \
    libjpeg-dev                          \
    libsdl2-dev                          \
    python3-dev                          \
    python3-opengl                       \
    swig                                 \
    xorg-dev                             \
    xvfb                                 \
    zlib1g-dev

# OpenBLAS & various dependencies
RUN cd /tmp                                             \
    && apt-get install -qy git-core nodejs-legacy npm   \
    && git clone https://github.com/xianyi/OpenBLAS.git \
    && npm install -g configurable-http-proxy           \
    && cd OpenBLAS                                      \
    && make DYNAMIC_ARCH=1 NO_AFFINITY=1 NUM_THREADS=32 \
    && make PREFIX=/opt/openblas install                \
    && pip3 install --upgrade numpy                     \
    && mkdir /usr/bin/gcc-for-nvcc                      \
    && ln -s /usr/bin/gcc-4.9 /usr/bin/gcc-for-nvcc/gcc \
    && echo "compiler-bindir = /usr/bin/gcc-for-nvcc/" >> /usr/local/cuda/bin/nvcc.profile \
    && mkdir /tmp/.X11-unix                             \
    && chmod 1777 /tmp/.X11-unix

# Rltut module & examples
COPY requirements.txt /tmp/rltut/
RUN pip3 install -r /tmp/rltut/requirements.txt

COPY . /tmp/rltut
RUN cp -r /tmp/rltut/examples /examples                    \
    && mkdir -p /etc/jupyterhub/                           \
    && cp /tmp/rltut/jupyterhub_config.py /etc/jupyterhub/ \
    && rm -r /tmp/rltut
