# Evasive Acceleration (EA)

A two-dimensional paradigm for instantaneous driving risk quantification.

## Table of Contents
- [Visual Overview](#visual-overview)
- [Overview](#overview)
- [Why EA?](#why-ea)
- [Installation & Requirements](#installation--requirements)
- [Repository Structure](#repository-structure)
- [Input Definition](#input-definition)
- [Usage](#usage)
  - [1. Single-Frame Computation](#1-single-frame-computation)
  - [2. Batch Computation](#2-batch-computation)
  - [3. Visualization](#3-visualization)
- [What the Code Computes](#what-the-code-computes)
- [License](#license)

---

## Visual Overview

### Naturalistic Interactions
<p align="center">
  <img src="assets/gifs/SIND_Tianjin_8_9_1_486_496.gif" alt="SinD Tianjin case" width="90%">
</p>
A representative high-risk intersection interaction from the SIND dataset, involving one vehicle and one motorcycle.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

### Reconstructed Crash Cases
<p align="center">
  <img src="assets/gifs/crash_684_T-bone.gif" alt="Crash 684 T-bone" width="90%">
</p>
A T-bone collision case from the CIMSS-TA database.   
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

---

## Overview

Evasive Acceleration (EA) quantifies driving risk as the minimum constant relative acceleration required to make a predicted interaction collision-free. Unlike time-to-collision (TTC)-based methods, EA evaluates risk in a two-dimensional joint-motion space rather than through one-dimensional temporal proximity alone.

This repository provides an implementation of EA for two practical use cases:
1. **Single-frame computation**: For real-time or instant-evaluation scenarios, such as online risk monitoring and reinforcement learning.
2. **Batch computation**: For offline processing of trajectory datasets, where EA and baseline metrics are computed frame by frame from CSV files.

---

## Why EA?

Most existing risk metrics in autonomous driving are built around the TTC paradigm. These methods quantify risk mainly through a single temporal dimension, even though real traffic interactions are inherently two-dimensional.

This mismatch can lead to two common issues:
- Interactions with substantially different avoidance difficulty may receive similar TTC-like values.
- Risk may appear to keep increasing even after an effective evasive manoeuvre has already started resolving the conflict.

EA addresses this by measuring the minimum instantaneous intervention required to make the interaction collision-free. It reframes risk as the physical effort needed to restore safety, rather than as time remaining under nominal motion.

---

## Installation & Requirements

This repository is written in Python. It is recommended to run the code within a virtual environment.

### Core Libraries
The codebase requires the following dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`

### Optional Acceleration
We highly recommend installing `numba`. If `numba` is installed, some core computations will be significantly accelerated. The code will still run without it, but execution speed will be reduced.

### Quick Install
You can install all required and recommended libraries via pip:

```bash
pip install numpy pandas matplotlib pillow numba
```

---

## Repository Structure

To keep things simple, here are the core components you need to care about:

```text
evasive-acceleration/
├── demo_data/                # Example CSV data for quick batch testing
├── src/
│   ├── core_ea.py            # Core EA solver and motion-mode evaluations
│   ├── baseline_risk_metrics.py  # TTC, ACT, DRAC, MEI implementations
│   ├── single_frame.py       # Entry script for instant single-frame evaluation
│   └── batch_compute.py      # Entry script for processing full trajectory datasets
└── visualization/            # Scripts for rendering cases and exporting GIFs
```

---

## Input Definition

EA is computed from the instantaneous states of two road users. Each road user is represented by **7 parameters**.

### Required 7 Parameters
For road user i in {A, B}, the input is:
`(x_i, y_i, v_i, h_i, L_i, W_i, ω_i)`

where:
- `x_i`: global x position `[m]`
- `y_i`: global y position `[m]`
- `v_i`: speed magnitude `[m/s]`
- `h_i`: heading angle `[rad]`
- `L_i`: body length `[m]`
- `W_i`: body width `[m]`
- `ω_i`: yaw rate `[rad/s]`

So one interaction frame consists of **14 values in total**.

**Command-Line Order:** `x y speed heading length width yaw_rate`

### Notes on Yaw Rate Input
Some trajectory datasets do not provide yaw rate directly. You can estimate it from the historical heading sequence by a finite-difference step (smoothing recommended).
- If both road users do not exhibit noticeable turning behaviour, you may set `yaw_rate = 0`.
- If either road user is clearly turning, it is highly recommended to provide an accurate yaw-rate input.

### Applicability to Vulnerable Road Users (VRUs)
The method can be applied directly to cyclists or pedestrians. The input format remains the same; simply adjust the `length` and `width` to reflect the physical size of the VRU.

---

## Usage

### 1. Single-Frame Computation
Use `src/single_frame.py` for real-time EA evaluation, reinforcement learning, or instant inspection of a single state.

**Run the Built-In Example:**
```bash
python src/single_frame.py
```

**Run With Your Own Input:**
```bash
python src/single_frame.py \
  --agent-a 0 0 10 0 4.5 1.8 0 \
  --agent-b 20 0 8 3.1415926 4.7 1.9 0
```
*(Agent A is at (0,0) moving at 10m/s. Agent B is at (20,0) moving at 8m/s in the opposite direction).*

### 2. Batch Computation
Use `src/batch_compute.py` to process CSV files or trajectory cases frame by frame. Ideal for dataset analysis and comparing EA with baselines.

**Run Batch Computation:**
```bash
python src/batch_compute.py
```
**Typical Workflow:**
1. Place your input CSV files in `demo_data/` (or your target directory).
2. Check the input/output path settings inside `src/batch_compute.py`.
3. Run the script and inspect the generated output files.

### 3. Visualization
Use the scripts in `visualization/` to render interaction geometry and export GIFs.
1. Prepare the corresponding CSV case.
2. Run the target visualization script.
3. Inspect the generated images or GIFs.

---

## What the Code Computes

### EA
The final EA value is computed as the arithmetic mean of four motion-mode-specific values (`EA_CTCT`, `EA_CTCV`, `EA_CVCT`, `EA_CVCV`), which correspond to different short-horizon nominal motion assumptions.

### Baseline Metrics
The repository includes several baseline risk metrics for comparison: `TTC`, `TTC2D`, `ACT`, `DRAC`, and `MEI`. These are implemented separately in `baseline_risk_metrics.py`.

---

## License

This project is released under the license provided in `LICENSE`.
