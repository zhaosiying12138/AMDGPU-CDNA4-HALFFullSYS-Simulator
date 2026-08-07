# Evidence directory

Each accepted checkpoint adds a machine-readable manifest under
`state/evidence/`. Large logs, traces, binaries, and model files stay outside
Git; manifests record their size/SHA-256 and generating command.
