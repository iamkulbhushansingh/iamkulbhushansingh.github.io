---
layout: page
title: Optical Flow + Kalman/RANSAC Fusion
description: Robust motion estimation pipeline fusing Lucas-Kanade, Farneback, Horn-Schunck with Kalman filter and RANSAC
img:
importance: 4
category: work
---

Multi-algorithm optical flow pipeline with robust outlier rejection for video analysis applications.

**Algorithms implemented:**
- Lucas-Kanade sparse optical flow (feature point tracking)
- Farneback dense optical flow (per-pixel motion field)
- Horn-Schunck variational optical flow

**Fusion layer:**
- **Kalman filter** — temporal smoothing of flow estimates across frames
- **RANSAC** — rejection of flow vectors inconsistent with dominant motion model
- Affine/homography model fitting for global motion estimation

**Applications:**
- Camera motion compensation for VQA
- Motion vector validation for codec artifact detection
- Global motion estimation for scene classification

**Stack:** C++17 · OpenCV · SIMD intrinsics

Performance: 40–60% speedup over baseline OpenCV calls via SIMD-accelerated inner loops.
