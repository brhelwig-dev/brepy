FROM ghcr.io/astral-sh/uv:0.13.7-python3.14-trixie-slim@sha256:e80c05a310400cfc90af00b5b7271ef9ad24816e3265e1162af8a678e1aad099
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN uv export --project /src --locked --no-hashes --no-emit-project --format requirements-txt > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt /src \
    && rm -rf /src /tmp/requirements.txt
