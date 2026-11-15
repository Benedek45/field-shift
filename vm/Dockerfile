FROM ubuntu:24.04

ARG OPENCODE_SHA256

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl jq ripgrep git \
    && rm -rf /var/lib/apt/lists/*

COPY opencode-linux-arm64.tar.gz /tmp/opencode.tar.gz
RUN echo "${OPENCODE_SHA256}  /tmp/opencode.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin \
    && rm /tmp/opencode.tar.gz \
    && chmod 755 /usr/local/bin/opencode

RUN useradd --create-home --uid 1001 advisor \
    && mkdir -p /work /cfg /data \
    && chown advisor:advisor /work /data
USER advisor
ENV HOME=/home/advisor XDG_CONFIG_HOME=/cfg XDG_DATA_HOME=/data
WORKDIR /work
