---
layout: page
title: Video Dropout Detection
description: Patent-filed system for detecting chroma dropout, blocky dropout, and concealment artifacts
img: assets/img/3.jpg
importance: 2
category: work
---

Production artifact detection system covering multiple classes of video dropout, filed as a patent with me as 1st inventor.

**Artifact types covered:**
- **Chroma dropout** — detection of color channel loss across frames
- **Blocky dropout** — DCT block boundary artifacts from codec failure or packet loss
- **Concealment artifacts** — error concealment patterns from H.264/AVC decoder fallback
- **Black bar detection** — letterbox/pillarbox bars including 16-bit HDR and 4K content

**Approach:**
- Spatial frequency analysis for blockiness quantification
- Chroma plane statistical deviation tracking
- Reference-free (no original content required)
- Configurable sensitivity thresholds per content class

**Stack:** C++17 · FFmpeg · Intel intrinsics (AVX2/SSE4)

Deployed in commercial broadcast monitoring products.
