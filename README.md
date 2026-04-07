# Evasive Acceleration (EA)

A two-dimensional paradigm for instantaneous driving risk quantification.

> [!NOTE]
> **Visual overview placeholders**
>
> - **Main GIF (recommended):** a representative merging case where EA rises before effective yielding and then declines after the evasive manoeuvre takes effect.
> - **Secondary GIF:** a representative crash case where EA increases continuously until impact.
> - **Acceleration-space GIF:** a visualization of the collision-avoiding set and the EA vector in two-dimensional relative-acceleration space.
>
> You can place these assets under `assets/gifs/` and embed them here later.

---

## Overview

Evasive Acceleration (EA) quantifies driving risk as the minimum constant relative acceleration required to make a predicted interaction collision-free over a finite horizon. Unlike time-to-collision (TTC)-based methods, EA evaluates risk in a two-dimensional joint-motion space rather than through one-dimensional temporal proximity.

This repository provides the research implementation of EA, including:

- the core numerical solver for EA under multiple motion-model combinations;
- the analytical CV-based formulation and a set of baseline risk metrics;
- batch computation from CSV files;
- example data and visualization utilities.

The repository is designed for two purposes:

1. to help users understand why EA differs fundamentally from TTC-based methods;
2. to provide a clear and reproducible implementation of EA computation.

---

## Why EA?

Most existing risk metrics in autonomous driving are built on the TTC paradigm. TTC-based methods quantify risk through a single temporal dimension, even though traffic interactions are inherently two-dimensional and multidirectional.

This mismatch can lead to two common problems:

- interactions with substantially different avoidance difficulty may receive similar TTC values;
- risk may continue to appear to increase even after an effective evasive manoeuvre has already started to resolve the conflict.

EA addresses this problem by measuring the minimum instantaneous intervention required for collision avoidance. It reframes risk as the two-dimensional cost of remaining collision-free, rather than as time remaining under nominal motion.

---

## What this repository contains

- `src/core_ea.py`  
  Core implementation of EA. This file contains the main numerical solver, the four motion-mode evaluations, and the final model-averaged EA computation.

- `src/analytical_core.py`  
  Analytical CV-based EA and baseline risk metrics, including TTC, DRAC, TAdv, ACT, TTC2D, EI, and MEI.

- `src/batch_compute.py`  
  Batch computation over CSV files. This script reads frame-wise interaction data, computes EA and analytical metrics, and writes the results back to new CSV files.

- `examples/`  
  Minimal runnable examples for single-frame evaluation, batch processing, and visualization.

- `demo_data/`  
  Small example CSV files for quick testing.

- `tools/`  
  Utilities for visualization and asset generation.

- `assets/`  
  GIFs and figures used in this README.

- `docs/`  
  Short supporting documentation on data format, output fields, and method overview.

---

## Repository structure

```text
evasive-acceleration/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   ├── gifs/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── core_ea.py
│   ├── analytical_core.py
│   └── batch_compute.py
├── examples/
│   ├── single_frame_demo.py
│   ├── batch_demo.py
│   └── visualize_demo.py
├── demo_data/
│   ├── input/
│   └── output/
├── tools/
│   ├── visualize_csv_case.py
│   ├── make_gif_case.py
│   └── export_readme_assets.py
└── docs/
    ├── method_overview.md
    ├── data_format.md
    └── output_fields.md
