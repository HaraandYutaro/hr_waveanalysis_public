"""Dataset loading layer for the SEG-2 / JSON sidecar input architecture.

This package is introduced as the canonical home for orchestration
entry points that combine immutable raw sources (``.sg2`` files) with
derived sidecar metadata (``.json`` files) and derived cached
snapshots (``.npz`` files). See ``private/docs/ai_refactor.md`` §3
and §4 for the layered responsibility model.

Contents:

- :mod:`src.io.dataset_loader` — public loader entry points
  (currently only :func:`~src.io.dataset_loader.load_from_sg2_json`).
"""
