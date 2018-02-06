FROM chainer/chainer:v4.0.0b3-python3

RUN apt-get update && apt-get install -y \
    ffmpeg                               \
    python3-opengl                       \
    xvfb

COPY . /tmp/rltut
RUN pip3 install -r /tmp/rltut/requirements.txt

RUN mkdir /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix
