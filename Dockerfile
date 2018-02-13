FROM chainer/chainer:v4.0.0b3-python3

RUN apt-get update && apt-get install -y \
    cmake                                \
    ffmpeg                               \
    libav-tools                          \
    libboost-all-dev                     \
    libjpeg-dev                          \
    libsdl2-dev                          \
    python3-dev                          \
    python3-opengl                       \
    swig                                 \
    xorg-dev                             \
    xvfb                                 \
    zlib1g-dev


RUN apt-get install -y libffi-dev

COPY . /tmp/rltut
RUN pip3 install -r /tmp/rltut/requirements.txt

RUN mkdir /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix
