FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash ca-certificates coreutils findutils grep iproute2 \
       procps python3 tar xz-utils zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/zhaosiying/amdgpu-sim

CMD ["/bin/bash"]
