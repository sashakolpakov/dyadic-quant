FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG OLLAMA_ARCH=amd64
RUN curl -fsSL "https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.zst" \
      -o /tmp/ollama.tar.zst \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
    && rm /tmp/ollama.tar.zst

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch \
    && python3 -m pip install -r /workspace/requirements.txt

COPY . /workspace
RUN chmod +x /workspace/dyadic-experiments.sh /workspace/scripts/docker-entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/workspace/scripts/docker-entrypoint.sh"]
