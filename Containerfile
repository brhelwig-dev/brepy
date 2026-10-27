FROM ghcr.io/astral-sh/uv:0.13.6-python3.14-trixie-slim@sha256:637619278855a77885a94312efa7decd8479ca6cec5f959e134cbab96582821e
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN uv export --project /src --locked --no-hashes --no-emit-project --format requirements-txt > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt /src \
    && rm -rf /src /tmp/requirements.txt
