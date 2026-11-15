FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl jq ripgrep git \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
# Official installer uses these npm archives. SHA-512 pins are dist.integrity
# from registry.npmjs.org/@opencode/cli-linux-{arm64,x64-baseline}/2.3.0.
# Use x64-baseline so older amd64 servers do not require AVX2.
RUN set -eu; \
    case "${TARGETARCH}" in \
      arm64) target=arm64; checksum=d48175350143e66759ecd714f355227dd60054a10300ffc9d3bca56e01386404e68a74bba4e9f948e5d5cf23a4b8dd562dc0f35ee9e48eed21c3c86e838f304e ;; \
      amd64) target=x64-baseline; checksum=c08328f5a9d42485659c126518cd2507b9fe8e614cb2d4708cde8bc3cfacd21a7e1e903de0b9953030e44e0fa242acdcb1ab6525a1eb7b7f6ff31fa319012b45 ;; \
      *) echo "Unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fSL "https://registry.npmjs.org/@opencode/cli-linux-${target}/-/cli-linux-${target}-2.3.0.tgz" -o /tmp/opencode.tar.gz; \
    echo "${checksum}  /tmp/opencode.tar.gz" | sha512sum -c - \
    && tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin --strip-components=2 package/bin/opencode \
    && rm /tmp/opencode.tar.gz \
    && chmod 755 /usr/local/bin/opencode
RUN version="$(opencode --version)" \
    && printf 'OpenCode version: <%s>\n' "$version" \
    && test "$version" = "opencode v2.3.0"

RUN useradd --create-home --uid 1001 advisor \
    && mkdir -p /work /cfg /data \
    && chown advisor:advisor /work /cfg /data
USER advisor
ENV HOME=/home/advisor XDG_CONFIG_HOME=/cfg XDG_DATA_HOME=/data
WORKDIR /work
RUN test "$(opencode --version)" = "opencode v2.3.0"
