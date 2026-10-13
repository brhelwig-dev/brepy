FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:8e70ac2e85d09a5bb7ef09753b80671a60a71c8d24f195e4890d71e03865e36d
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN uv export --project /src --locked --no-hashes --no-emit-project --format requirements-txt > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt /src \
    && rm -rf /src /tmp/requirements.txt
