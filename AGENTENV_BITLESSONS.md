# AgentENV Integration Lessons

1. AgentENV's ublk-backed disks and snapshots make the host kernel a hard prerequisite; `/dev/kvm` alone is insufficient.
2. AgentENV directory upload is not a valid runtime-environment transport for this project because it rejects symlinks and does not preserve all executable metadata. Bundle a tar.zst and extract inside the guest.
3. The guest must remain Ubuntu 26.04 for the current gem5 binary's glibc requirement; an Ubuntu 24.04 guest is not interchangeable.
4. Active runtime manifests contain absolute references and symlink closures. A bundle builder must recursively include referenced prefixes instead of copying only the visible active directory.
5. AgentENV's current API has no authentication. Bind its service to loopback and put instance control behind the feature-local manager.
6. A common snapshot is safe only when both sandboxes use identical fixed resources. Per-instance state must still be separated at VM, PID, filesystem, network, tmp, cache, socket, log, SMI, and endpoint levels.
7. A custom WSL kernel switch is globally disruptive. Build and stage artifacts first; never hide `wsl --shutdown` inside a script.
8. vLLM/SGLang launch failures after the existing initialization marker are evidence, not a reason to widen this environment integration into model debugging.
9. AgentENV warm `templateID` creation does not accept CPU, memory, or disk overrides; those values must be fixed by the template (or use the cold API). Metadata alone is a report/verification contract, not resource enforcement.
10. Pair creation must be transactional: if a later sandbox POST or local state write fails, delete every sandbox already created and mark its local record deleted. A host-only lifecycle test now covers this rollback.
11. Public source pins must use HTTPS rather than host SSH credentials. A fresh
    AgentENV guest must be able to initialize the pinned sources without
    inheriting the host's private key or agent socket.
12. WSL 2.7.10 overlays the root of a custom `kernelModules` VHD directly on
    `/usr/lib/modules/<kernelrelease>`, but the pinned 6.18 generator targets a
    newer `<kernelrelease>/{modules,linux-headers,perf}` contract. The nested
    layout alone makes `mini_init` module loads fail on 2.7.10. Preserve it for
    newer runtimes and add relative root aliases (`modules.dep`, `kernel/`,
    ...) for older runtimes; this is dual-compatible without duplicating the
    module payload.
