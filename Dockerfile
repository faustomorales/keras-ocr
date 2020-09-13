FROM python:3.7
WORKDIR /usr/src
# Avoid re-downloading weights when other things
# change.
RUN mkdir -p /root/.keras-ocr && ( \
    cd /root/.keras-ocr && \
    curl -L -o craft_mlt_25k.h5 https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.h5 && \
    curl -L -o crnn_kurapan.h5 https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/crnn_kurapan.h5 \
    )
COPY ./Pipfile* ./
COPY ./Makefile ./
COPY ./setup* ./
COPY ./versioneer* ./
COPY ./docs/requirements.txt ./docs/requirements.txt
RUN pip install pipenv && make init
ENV LC_ALL C