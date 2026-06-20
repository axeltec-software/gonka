# syntax=docker/dockerfile:1.4
################################################################################
# vLLM base image: our v0.20 decode-PoC engine (torch 2.11 / cu128, sm_80/89/90).
ARG VLLM_BASE=ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128
FROM ${VLLM_BASE} AS base

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-pip \
    build-essential \
    checkinstall \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade "setuptools<81" wheel pip \
    && curl -fLO https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1u/openssl-1.1.1u.tar.gz \
    && tar -xzf openssl-1.1.1u.tar.gz \
    && cd openssl-1.1.1u \
    && ./config --prefix=/usr/local/openssl-1.1 \
                --openssldir=/usr/local/openssl-1.1/ssl \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf openssl-1.1.1u.tar.gz openssl-1.1.1u

ENV PATH="/usr/local/openssl-1.1/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/openssl-1.1/lib:$LD_LIBRARY_PATH"

################################################################################
FROM base AS builder

ENV POETRY_VERSION=2.0.1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    DEBIAN_FRONTEND=noninteractive

# Optional extra flash-attn wheel for the api venv. Empty by default: the v0.20 base
# is torch 2.11 (the old torch2.9 wheel would be ABI-mismatched), mlnode python does
# not import flash_attn, and the engine runs FlashInfer at runtime. Set a torch-2.11
# wheel URL here only if a future need arises.
ARG FLASH_ATTN_WHEEL=""

RUN pip install --upgrade pip && \
    pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY packages/pow/pyproject.toml packages/pow/poetry.lock packages/pow/README.md \
    /app/packages/pow/
COPY packages/train/pyproject.toml packages/train/poetry.lock packages/train/README.md \
    /app/packages/train/
COPY packages/api/pyproject.toml packages/api/poetry.lock packages/api/README.md \
    /app/packages/api/
COPY packages/common/pyproject.toml packages/common/poetry.lock packages/common/README.md \
    /app/packages/common/

COPY packages/train/third_party /app/packages/train/third_party

RUN mkdir -p /app/packages/api/src/api \
        && touch /app/packages/api/src/api/__init__.py \
    && mkdir -p /app/packages/pow/src/pow \
        && touch /app/packages/pow/src/pow/__init__.py \
    && mkdir -p /app/packages/train/src/zeroband \
        && touch /app/packages/train/src/zeroband/__init__.py \
    && mkdir -p /app/packages/common/src/common \
        && touch /app/packages/common/src/common/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    python3 -m venv /app/packages/api/.venv \
    && . /app/packages/api/.venv/bin/activate \
    && cd /app/packages/api \
    && poetry install --no-interaction \
    && if [ -n "$FLASH_ATTN_WHEEL" ]; then pip install "$FLASH_ATTN_WHEEL"; fi

################################################################################
FROM base AS app
COPY --from=builder /app/packages/api/.venv /app/packages/api/.venv

COPY packages/api/src /app/packages/api/src

COPY packages/pow/src /app/packages/pow/src
COPY packages/pow/resources /app/packages/pow/resources
COPY packages/pow/scripts /app/packages/pow/scripts

COPY packages/train/src /app/packages/train/src
COPY packages/train/scripts /app/packages/train/scripts
COPY packages/train/resources /app/packages/train/resources

COPY packages/common/src /app/packages/common/src

ENV PYTHONPATH="/app:/app/packages/api/src"
ENV VLLM_ALLOW_INSECURE_SERIALIZATION=1
ENV LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

COPY packages/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]


################################################################################
FROM builder AS test

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    cd /app/packages/api/ \
    && poetry install --with dev \
    && . /app/packages/api/.venv/bin/activate

COPY packages/api/src /app/packages/api/src
COPY packages/pow/src /app/packages/pow/src
COPY packages/pow/resources /app/packages/pow/resources
COPY packages/pow/scripts /app/packages/pow/scripts

COPY packages/train/src /app/packages/train/src
COPY packages/train/scripts /app/packages/train/scripts
COPY packages/train/resources /app/packages/train/resources

COPY packages/common/src /app/packages/common/src

COPY packages/api/tests /app/packages/api/tests
COPY packages/pow/tests /app/packages/pow/tests
COPY packages/train/tests /app/packages/train/tests

ENV PYTHONPATH="${PYTHONPATH}:/app/packages/api/src"

COPY packages/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]