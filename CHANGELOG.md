# Changelog

## 2026-10-13 — Portfolio and verification update

- Add a visual README, architecture/workflow/hardware guides and explicit result provenance.
- Add four reproducible PNG/SVG charts: reported accuracy, HLS resources, model workload and latency scopes.
- Extract HLS metrics with the raw artifact's SHA-256 and calculate model complexity from layer dimensions.
- Support optional plots from actual training history; do not invent missing curves or distributions.
- Restore C++ testbench coverage of all golden samples and sample-0 replay.
- Add offline NumPy/C++ checks, an intentional-mismatch test and Python 3.11/3.12 CI.
- Preserve existing trained parameters, vectors, accelerator implementation and archived AMD artifacts.
