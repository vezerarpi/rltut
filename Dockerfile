FROM chainer/chainer:v4.0.0b3-python3

COPY . /tmp/rltut
RUN pip3 install -r /tmp/rltut/requirements.txt
