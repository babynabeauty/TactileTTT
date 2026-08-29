FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y \
    git curl wget build-essential ffmpeg libgl1 libglib2.0-0 \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/FactileLDM

COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY src ./src
COPY scripts ./scripts
COPY README.md LICENSE ./

RUN uv sync --frozen

COPY . .

CMD ["/bin/bash"]

