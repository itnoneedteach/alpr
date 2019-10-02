FROM ubuntu:18.04

# install tools
RUN apt-get update \
  && apt-get install -y wget libtbb2 libtbb-dev libavcodec57 libavformat57 libavutil55 libcairo2 libgdk-pixbuf2.0-0 libglib2.0-0 libgstreamer-plugins-base1.0-0 libgstreamer1.0-0 libgtk2.0-0 libjpeg8 libpng16-16 libswscale4 libtiff5 python python-pip pkg-config libsm6

COPY . /data

WORKDIR /data

RUN cd /data/darknet && make -j 4

RUN python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose keras tensorflow opencv-python exifread python-telegram-bot

RUN bash get-networks.sh

CMD python /data/int.py
