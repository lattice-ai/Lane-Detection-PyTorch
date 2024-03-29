# Use the CUDA image as base
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# metainformation
LABEL version="0.0.1"
LABEL maintainer="Saurav Maheshkar"
LABEL org.opencontainers.image.source = "https://github.com/SauravMaheshkar/Lane-Detection-PyTorch"

# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /code

# Essential Installs
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		gfortran \
		libopenblas-dev \
		python3 \
		python3-pip \
		python3-dev \
		python3-venv \
		&& apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel isort
RUN python3 -m pip install --no-cache-dir -r requirements.txt
