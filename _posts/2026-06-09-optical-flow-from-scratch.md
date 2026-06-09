---
layout: post
title: "Optical flow from scratch: the solve is trivial, the box filter is everything"
date: 2026-06-09 10:00:00 +0530
description: "Dense Lucas-Kanade from first principles — brightness constancy, the aperture problem, the 2×2 solve, and why swapping one loop for a box filter gives 250× speedup with identical output."
tags: [computer-vision, optical-flow, lucas-kanade, image-processing, python, performance]
categories: [math-under-the-pixels]
thumbnail: /assets/img/blog/optical-flow/flow_hero.png
featured: true
---

> **512×512 image, 5.66 px max motion, zero neural net.**
> Single-scale Lucas-Kanade: EPE **2.792 px**. Add a four-level pyramid: **0.752 px**.
> Change one function call (loop → box filter): **250× faster**. Same output.

That's the whole post. Everything below is *why*.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/flow_hero.png" class="img-fluid rounded" zoomable=true caption="Dense flow on the camera image. Hue = direction of motion, saturation = speed. Every pixel has a vector. No model, no GPU." %}
  </div>
</div>

---

## What is optical flow?

Two frames of video. For every pixel in frame 1, find where it moved in frame 2. Output: a 2D vector field — one `(u, v)` per pixel.

That's it. No segmentation, no depth, no semantics. Just: *where did each point go?*

This is the front end of:
- **Visual odometry / SLAM** — track points, recover camera pose
- **ADAS / drones** — ego-motion, moving-obstacle detection, time-to-collision
- **Video codecs** — motion estimation inside every H.264/AV1 encoder
- **Video stabilization** — estimate camera shake, warp it out

All of them need it fast and predictable. That's the engineering problem.

---

## Step 1 — what does the algorithm actually see?

Before any math: look at what you're working with.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig1_gradients.png" class="img-fluid rounded" zoomable=true caption="Frame 1, Frame 2, and the three gradient images Iₓ, I_y, I_t. The algorithm never touches colour — just these." %}
  </div>
</div>

The three gradient images are the raw ingredients:

- **Iₓ** — horizontal intensity change. Red/blue = sign of the change.
- **I_y** — vertical intensity change.
- **I_t** — temporal change (frame2 − frame1). Where pixels moved, you see signal here.

The entire optical flow algorithm is built from these three images. That's all it has.

---

## Step 2 — the constraint: brightness constancy

The founding assumption: *a pixel keeps its intensity as it moves.*

```
I(x, y, t)  =  I(x + u, y + v, t + 1)
```

Expand the right side with a first-order Taylor series:

```
I(x+u, y+v, t+1)  ≈  I(x,y,t) + Iₓ·u + I_y·v + I_t
```

Substitute and the `I(x,y,t)` cancels:

```
Iₓ·u  +  I_y·v  +  I_t  =  0          … (1)
```

This is the **optical flow constraint equation**. One equation. Two unknowns (`u`, `v`).

### The aperture problem

One equation, two unknowns — you can't solve it. That's not a bug; it's geometry.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig2_aperture.png" class="img-fluid rounded" zoomable=true caption="Edge patch: all red arrows satisfy the constraint. The algorithm genuinely cannot pick one. Corner patch: two independent edges cross — only one arrow fits." %}
  </div>
</div>

Through a small window over an edge, you can only measure motion *perpendicular* to that edge — the component along the edge is invisible. This is the **aperture problem**.

At a corner both `Iₓ` and `I_y` are large, so you get two independent constraints. Their intersection is a unique solution. This is why feature trackers pick corners.

---

## Step 3 — Lucas-Kanade: borrow from the neighbours

**Key assumption:** flow is approximately constant over a small `k×k` window.

Every pixel in the window gives one constraint equation. For a 15×15 window that's **225 equations, 2 unknowns** — overdetermined. No exact solution exists, so we minimize total squared error (least squares). Here's how 225 equations become 2×2.

### From 225 equations to a matrix

Write every constraint in the window as one row:

```
pixel 1:    Ix1·u + Iy1·v = -It1
pixel 2:    Ix2·u + Iy2·v = -It2
...
pixel 225:  Ix225·u + Iy225·v = -It225
```

Stack them into a matrix system `A·x = b`:

```
⎡ Ix1   Iy1  ⎤         ⎡ -It1  ⎤
⎢ Ix2   Iy2  ⎥  [u]  = ⎢ -It2  ⎥
⎢  ...       ⎥  [v]    ⎢  ...  ⎥
⎣ Ix225 Iy225⎦         ⎣ -It225⎦
    A (225×2)  · x  =    b (225×1)
```

### Least squares collapse

Minimize `‖Ax − b‖²`. Take the derivative with respect to `x`, set to zero:

```
AᵀA · x  =  Aᵀb
```

`AᵀA` is **(2×225)·(225×2) = 2×2**. Expand it:

```
AᵀA = ⎡ ΣIx²    ΣIx·Iy ⎤        Aᵀb = ⎡ ΣIx·It ⎤
      ⎣ ΣIx·Iy  ΣIy²   ⎦               ⎣ ΣIy·It ⎦
```

All 225 equations are still in there — compressed into five window sums. The full system:

```
⎡ ΣIₓ²    ΣIₓI_y ⎤ ⎡u⎤     ⎡ ΣIₓI_t ⎤
⎢                  ⎥ ⎢ ⎥  = -⎢         ⎥
⎣ ΣIₓI_y  ΣI_y²  ⎦ ⎣v⎦     ⎣ ΣI_yI_t ⎦

       A               [u,v]        b
```

Solve with Cramer's rule — two divisions. Done.

The sums run over the `k×k` window. `A` is the **structure tensor** — the same matrix Harris and Shi-Tomasi use to find corners. Its smallest eigenvalue tells you how well the patch constrains flow:

| min eigenvalue | meaning |
|:---:|---|
| ≈ 0 | flat region — no texture, flow unknowable |
| one large, one ≈ 0 | edge — aperture problem |
| both large | corner — fully determined, trackable |

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig3_structure_tensor.png" class="img-fluid rounded" zoomable=true caption="Min-eigenvalue map. Bright = trackable. Most pixels live in the hopeless flat zone — the histogram's spike at zero confirms it." %}
  </div>
</div>

### Regularization instead of a hard reject

When `A` is singular, the solve blows up. The typical fix is a threshold on the determinant — but that creates a hard discontinuity and silently zeroes marginal patches.

Better: add a tiny Tikhonov term `λI` to `A`:

```
(A + λI) · [u,v]  =  -b        λ = 1e-9
```

Now `A + λI` is always invertible. In flat regions the flow decays to zero (honest "I don't know") instead of crashing. The `λ` is small enough to not bias results where texture exists.

---

## Step 4 — the performance trick (this is the whole point)

To compute the 2×2 system you need **five windowed sums**: `ΣIₓ²`, `ΣI_y²`, `ΣIₓI_y`, `ΣIₓI_t`, `ΣI_yI_t`.

**Naive:** loop over the `k×k` window per pixel. Cost: **O(k²) per pixel**.

For a 512×512 image with `k=15`: that's 512 × 512 × 225 ≈ **59 million additions**. Per sum. Five sums total.

**Key insight:** a sum over a sliding rectangular window *is a box filter* — and a box filter runs in **O(1) per pixel** via a prefix sum table, regardless of window size.

### How the prefix sum works

Pre-compute a table `P` where each cell = sum of everything above-and-left:

```
Original image:      Prefix sum table P:
  1  2  3              0   0   0   0
  4  5  6    →         0   1   3   6
  7  8  9              0   5  12  21
                       0  12  21  36
```

`P[y][x]` = sum of all values in the rectangle from `(0,0)` to `(y,x)`.

Now any rectangle sum = **4 lookups**, regardless of window size:

```
sum(r1,c1 → r2,c2)  =  P[r2][c2]
                      - P[r1-1][c2]
                      - P[r2][c1-1]
                      + P[r1-1][c1-1]   ← add back the corner subtracted twice
```

Visual:

```
+-------------------+
|        A          |
+--------+----------+
|   B    |  target  |
+--------+----------+

target = whole − A − B + corner
```

Build `P` once — O(N). Query every pixel — O(1). A 3×3 window and a 300×300 window both cost **4 lookups**.

Replace five nested loops with five filter calls:

```python
# Before — O(k²) per pixel, explicit loop
for y in range(r, h-r):
    for x in range(r, w-r):
        sxx = np.sum(Ix[y-r:y+r+1, x-r:x+r+1] ** 2)   # k² ops
        ...

# After — O(1) per pixel (uniform_filter uses prefix sums internally)
box  = lambda x: uniform_filter(x, size=win)
Sxx, Syy, Sxy = box(Ix*Ix), box(Iy*Iy), box(Ix*Iy)
Sxt, Syt      = box(Ix*It), box(Iy*It)
```

Same five products. Same windowed averages. Numerically identical output. The difference:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig8_speedup.png" class="img-fluid rounded" zoomable=true caption="Naive O(k²)/px vs box filter O(1)/px. The speedup is flat across window sizes — the box filter time is truly O(1)." %}
  </div>
</div>

| window | naive O(k²)/px | box filter O(1)/px | speedup |
|:---:|:---:|:---:|:---:|
| 9×9    | 0.187 s | 0.0007 s | **277×** |
| 15×15  | 0.183 s | 0.0007 s | **264×** |
| 25×25  | 0.165 s | 0.0008 s | **217×** |

The naive time is roughly **flat** across window sizes — you're still looping over a fixed crop. The box filter time is also flat — because it's `O(1)` regardless of window size.

> **Knowing the math is the optimization.** The 2×2 solve is cheap. The bottleneck is the five sums. The bottleneck has an O(1) solution. One function call.

---

## Step 5 — why single-scale fails and what pyramids do

The Taylor expansion is only valid for **sub-pixel motion**. On a 512×512 image with motion up to **5.66 px**:

- Single-scale LK: **EPE = 2.792 px** — clearly broken
- Pyramidal LK (4 levels): **EPE = 0.752 px** — 3.7× better

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig6_single_vs_pyramid.png" class="img-fluid rounded" zoomable=true caption="Single-scale (EPE 2.792px) vs pyramidal LK (EPE 0.752px) on the same motion field. The single-scale result is noisy and wrong; the pyramid recovers the true motion." %}
  </div>
</div>

### How the pyramid works

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig4_pyramid.png" class="img-fluid rounded" zoomable=true caption="Four pyramid levels. At the coarsest level (32×32) the 5.66px motion becomes sub-pixel — the Taylor approximation holds there." %}
  </div>
</div>

At the coarsest level (32×32), the full 5.66 px motion is now sub-pixel relative to that scale. LK works there. Then:

```
for each level l from coarse to fine:
    upsample flow estimate × 2
    warp Frame 1 by –flow  (align it toward Frame 2)
    run LK on the residual
    add residual to accumulated flow
    apply median filter  (flow is smooth; kills outlier vectors)
```

Each level only estimates the *residual* motion after the coarser level accounts for the bulk. Motion at each level stays small enough for the linear approximation.

**One sign to get right:** warp Frame 1 by **−flow**, not +flow, to align it toward Frame 2. Flip the sign and every refinement step *adds* error — I watched the flow diverge 2 → 5 → 10 → 20 px across pyramid levels before catching it.

---

## Step 6 — four motion patterns, one color wheel

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig5_patterns.png" class="img-fluid rounded" zoomable=true caption="Translation, rotation, zoom, shear — each produces a distinct colour signature. Zoom/divergence lights up the full colour wheel because motion points outward in every direction." %}
  </div>
</div>

| pattern | EPE | real-world appearance |
|---|:---:|---|
| Translation (u=+3, v=+1) | 0.525 px | pan shot, drone lateral |
| Rotation (ω=0.03 rad/frame) | 1.179 px | spinning object, camera yaw |
| Zoom / divergence (s=0.02) | 0.950 px | forward motion — ADAS ego-motion |
| Shear (∂u/∂y=0.04) | 1.158 px | parallax, conveyor belt |

---

## Step 7 — where does it actually fail?

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="/assets/img/blog/optical-flow/fig7_error_map.png" class="img-fluid rounded" zoomable=true caption="EPE heatmap. Bright = high error. Failure concentrated at occlusion boundaries and textureless regions — both are structurally understood failure modes." %}
  </div>
</div>

Highest error at:
- **Occlusion boundaries** — the pixel being tracked disappears behind another surface. Brightness constancy breaks.
- **Textureless regions** — structure tensor is near-singular. LK decays to zero; if the GT is non-zero, that's error.

**Median EPE: 0.131 px.** Most pixels are tracked very accurately. The tail drags the mean.

These failure modes have documented classical fixes — forward-backward consistency check for occlusions, larger windows or spatial regularization for texture-less regions. No neural network needed.

---

## The math in one place

```
1. Compute gradients (central differences):
   Iₓ[y,x] = 0.5 * (I1[y, x+1] - I1[y, x-1])
   I_y[y,x] = 0.5 * (I1[y+1, x] - I1[y-1, x])
   I_t[y,x] = I2[y,x] - I1[y,x]

2. Form products:
   P1 = Iₓ², P2 = I_y², P3 = Iₓ·I_y, P4 = Iₓ·I_t, P5 = I_y·I_t

3. Windowed sum (box filter, O(1)/px):
   Sxx = box(P1),  Syy = box(P2),  Sxy = box(P3)
   Sxt = box(P4),  Syt = box(P5)

4. Solve (+ regularization):
   det = Sxx·Syy - Sxy² + λ            λ = 1e-9
   u   = -(Syy·Sxt - Sxy·Syt) / det
   v   = -(Sxx·Syt - Sxy·Sxt) / det
```

Four steps. Three of them are element-wise array ops. The fourth is the box filter.

---

## Where this goes: visual odometry

Dense flow gives you a motion vector per pixel. If you instead track *feature points* and recover the camera's 3D motion from those correspondences, you have **visual odometry** — the core of SLAM, drone navigation, and ADAS. The next post builds that pipeline: feature detection → LK tracking → essential matrix → pose recovery → trajectory on KITTI. This implementation feeds straight in.

---

## The fast version — where the real moat is

The Python `uniform_filter` already leans on compiled C with SIMD internally. The next rung is hand-written AVX2: 8 pixels per instruction, threaded over rows.

```cpp
// Sxx, Syy, Sxy, Sxt, Syt are __m256 — 8 pixels per register
__m256 det  = _mm256_add_ps(
                  _mm256_fmsub_ps(Sxx, Syy, _mm256_mul_ps(Sxy, Sxy)),
                  reg);
__m256 inv  = _mm256_div_ps(one, det);
__m256 u    = _mm256_mul_ps(
                  _mm256_fnmadd_ps(Syy, Sxt, _mm256_mul_ps(Sxy, Syt)), inv);
__m256 v    = _mm256_mul_ps(
                  _mm256_fnmadd_ps(Sxx, Syt, _mm256_mul_ps(Sxy, Sxt)), inv);
```

That version, threaded over rows, runs real-time on 1080p. Python is for the figures; C++ is for the point.

---

## Run it yourself

Pure NumPy + SciPy — no OpenCV, no PyTorch, no pretrained weights:

```bash
git clone https://github.com/iamkulbhushansingh/image-math-teardowns
cd 06-optical-flow
pip install numpy scipy matplotlib scikit-image imageio

python optical_flow.py                         # synthetic self-test → EPE + figure
python optical_flow.py frame1.png frame2.png   # any two frames → colour flow field
python bench.py                                # naive vs box-filter speedup table
python figures.py                              # regenerate all 9 blog figures
```

Extract two frames from any clip:
```bash
ffmpeg -i clip.mp4 -vf "select=eq(n\,100)+eq(n\,101)" -vsync 0 frame%d.png
```

---

*Part of the [Math Under the Pixels](/blog/) series — classical CV, from the equations up.*
*Next: Features + RANSAC — robust matching when half your correspondences are garbage.*
