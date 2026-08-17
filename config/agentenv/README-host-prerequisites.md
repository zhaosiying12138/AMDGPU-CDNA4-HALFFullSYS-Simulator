# AgentENV host prerequisites

Three host-side conditions must hold before a template build can succeed. Each
was an observed failure, not a precaution.

## 1. Base image registry

`docker.io` and `gcr.io` are unreachable from this host; `quay.io` is reachable
directly. Template builds must therefore name a `quay.io` base. The Python
stack additionally requires **glibc >= 2.39**, because the pinned wheels are
`manylinux_2_39`; `quay.io/centos/centos:stream9` ships glibc 2.34 and fails
with "is not a supported wheel on this platform". `quay.io/fedora/fedora:42`
(glibc 2.41) works.

## 2. ublk device permissions

overlaybd creates one ublk device per runtime device. `/dev/ublk-control` is
group-accessible, but the per-device `/dev/ublkc*` nodes are created
`root:root 0600`. The AgentENV server runs as an unprivileged uid holding
ambient `CAP_NET_ADMIN`/`CAP_SYS_ADMIN`, and neither capability bypasses DAC on
a device node, so the build fails with:

```
create new exclusive ublk device on pool miss: build ublk dev:
try open "/dev/ublkc0" ... PermissionDenied
```

Install `99-agentenv-ublk.rules` from this directory:

```sh
sudo cp config/agentenv/99-agentenv-ublk.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=ublk_char --action=change
```

Edit the `GROUP=` value if the operator account is not `zhaosiying`.

## 3. Unix socket path length

firecracker binds `<AENV_HOME_PATH>/firecracker-work/agentenv-fc-XXXXXX/firecracker.socket`.
`sockaddr_un.sun_path` is 108 bytes on Linux and the feature-local state root
under this worktree already spends 65 of them, so the bind failed with
`RunWithApi(FailedToBindAndRunHttpServer(... "path must be shorter than
SUN_LEN"))`. `tools/agentenv_service.py` now publishes a short
`/tmp/aenv-<uid>-<tag>-<hash>` symlink as `AENV_HOME_PATH` and
`AENV_RUNTIME_PATH`; every artefact still lives in the worktree.

## Starting the service

The capability wrapper runs its own `sudo -E`, which needs a tty, so invoke it
already-root. Use `/usr/bin/python3`: the conda 3.14 interpreter lacks
`os.pidfd_open`, which the ownership record requires.

```sh
sudo env AENV_RUN_USER="$(id -un)" \
  projects/AgentENV/scripts/run-with-capabilities.sh \
  /usr/bin/python3 tools/agentenv_service.py start
```

`stop` requires `--confirm`.
