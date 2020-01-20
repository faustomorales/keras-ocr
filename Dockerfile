FROM python:3.7
WORKDIR /usr/src
# Avoid re-downloading weights when other things
# change.
RUN mkdir -p /root/.keras-ocr && ( \
    cd /root/.keras-ocr && \
    curl -L -o craft_mlt_25k.pth https://www.mediafire.com/file/qh2ullnnywi320s/craft_mlt_25k.pth/file && \
    curl -L -o craft_mlt_25k.h5 https://www.mediafire.com/file/mepzf3sq7u7nve9/craft_mlt_25k.h5/file && \
    curl -L -o crnn_kurapan.h5 https://www.mediafire.com/file/pkj2p29b1f6fpil/crnn_kurapan.h5/file \
    )
COPY ./Pipfile* ./
COPY ./Makefile ./
COPY ./setup* ./
COPY ./versioneer* ./
COPY ./docs/requirements.txt ./docs/requirements.txt
RUN pip install pipenv && make init
ENV LC_ALL C