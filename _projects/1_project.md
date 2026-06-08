---
layout: page
title: Scene-Change Detection
description: Real-time detector achieving 97% precision and 100% recall for broadcast video monitoring
img: assets/img/12.jpg
importance: 1
category: work
---

High-precision scene-change detection system built in C++ for broadcast and surveillance pipelines.

**Results:** 97% precision · 100% recall

**Approach:**

- Temporal difference analysis across decoded frame buffers
- Adaptive thresholding tuned per content type (sports, news, static camera)
- Histogram-based fast rejection path for uniform/low-motion scenes
- Sub-frame granularity for frame-accurate edit point detection

**Stack:** C++17 · FFmpeg · SIMD intrinsics

Production-deployed across broadcast monitoring customers for ad insertion, content indexing, and quality assurance.
