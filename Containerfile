FROM ghcr.io/astral-sh/uv:0.13.3-python3.14-trixie-slim@sha256:02f7e4ef3aeb9e5bf2c4ab7476b622c9ec9bfe62c9ded95074d82f99c50c1471
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN uv export --project /src --locked --no-hashes --no-emit-project --format requirements-txt > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt /src \
    && rm -rf /src /tmp/requirements.txt
