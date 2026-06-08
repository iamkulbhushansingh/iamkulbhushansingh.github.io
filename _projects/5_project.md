---
layout: page
title: TorchMetrics Open Source Contributions
description: VIF metric reduction support and classification metric NaN fix merged into Lightning-AI/torchmetrics
img: assets/img/1.jpg
importance: 5
category: open-source
---

Contributions merged into [TorchMetrics](https://github.com/Lightning-AI/torchmetrics), the standard metrics library for PyTorch Lightning.

---

**[PR #3226](https://github.com/Lightning-AI/torchmetrics/pull/3226) — VIF `reduction='none'` support**

Added per-sample reduction mode to the Visual Information Fidelity (VIF) perceptual image quality metric.

- Enables batch-level quality analysis returning one score per image instead of only a mean
- Critical for video frame-level quality profiling across a sequence
- Follows TorchMetrics reduction API contract used by all other metrics

---

**[PR #3196](https://github.com/Lightning-AI/torchmetrics/pull/3196) — Classification metric NaN fix**

Fixed incorrect NaN propagation in binary, multiclass, and multilabel classification metrics.

- Edge case: zero-sample batches or all-negative inputs produced silent NaN instead of defined behaviour
- Fix ensures deterministic outputs and prevents silent failures in training loops

---

**Stack:** Python · PyTorch · TorchMetrics
