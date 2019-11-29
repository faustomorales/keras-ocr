FROM python:3.7
WORKDIR /usr/src
COPY ./Pipfile* ./
COPY ./Makefile ./
COPY ./setup* ./
COPY ./versioneer* ./
COPY ./docs/requirements.txt ./docs/requirements.txt
RUN pip install pipenv && make init
ENV LC_ALL C
RUN mkdir -p /root/.keras-ocr
RUN ( \
    cd /root/.keras-ocr && \
    curl -O https://storage.googleapis.com/keras-ocr/craft_mlt_25k.pth && \
    curl -O https://storage.googleapis.com/keras-ocr/craft_mlt_25k.h5 && \
    curl -O https://storage.googleapis.com/keras-ocr/crnn_kurapan.h5 \
    )