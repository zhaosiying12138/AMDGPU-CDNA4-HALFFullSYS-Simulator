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
