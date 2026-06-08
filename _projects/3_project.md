---
layout: page
title: No-Reference Video Quality Assessment
description: VQA system for H.264, AV1, and MPEG2 streams using contrastive learning
img: assets/img/7.jpg
importance: 3
category: work
---

No-reference (NR) video quality assessment system that scores video quality without access to original source content.

**Codec coverage:** H.264 · AV1 · MPEG2

**Approach:**
- Contrastive learning on codec-distorted frame pairs
- Learned perceptual quality features robust to compression artifacts
- Per-frame and temporal quality scoring
- Integration with FFmpeg decode pipeline for real-time analysis

**Key differentiators:**
- No reference content needed (blind quality estimation)
- Codec-aware: separate feature spaces per codec family
- H.264 deep focus: P/B-frame artifact patterns, quantization noise

**Stack:** C++ · Python · PyTorch · FFmpeg

Deployed for broadcast QC, OTT stream monitoring, and surveillance pipeline health checks.
