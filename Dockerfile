FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
LABEL maintainer="Jourdan Rodrigues <thiagojourdan@gmail.com>"

WORKDIR /server/

RUN pip install --no-cache-dir "poetry==1.7.1" && \
    poetry config virtualenvs.create false

COPY pyproject.toml ./

RUN poetry install --no-root --no-cache --no-interaction

COPY . .
