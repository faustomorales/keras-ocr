FROM python:3.7

RUN apt update && apt install -y pkg-config libcairo2-dev

WORKDIR /usr/src
COPY ./Pipfile* ./
COPY ./Makefile ./
COPY ./setup* ./
COPY ./versioneer* ./
RUN pip install pipenv && make init
ENV LC_ALL C