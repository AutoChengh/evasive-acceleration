# Evasive Acceleration (EA)

A two-dimensional paradigm for instantaneous driving risk quantification.

---

## Visual overview

### Naturalistic interactions

<p align="center">
  <img src="assets/gifs/InD_05_tracks_266_267.gif" alt="InD case 266-267" width="90%">
</p>

A representative merging interaction.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

<p align="center">
  <img src="assets/gifs/SIND_Tianjin_8_9_1_486_496.gif" alt="SinD Tianjin case" width="90%">
</p>

A representative urban interaction from the SinD dataset.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

### Reconstructed crash cases

<p align="center">
  <img src="assets/gifs/crash614_sideswipe.gif" alt="Crash 614 sideswipe" width="90%">
</p>

A reconstructed sideswipe crash case.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

<p align="center">
  <img src="assets/gifs/crash_647_rear-end.gif" alt="Crash 647 rear-end" width="90%">
</p>

A reconstructed rear-end crash case.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

<p align="center">
  <img src="assets/gifs/crash_684_T-bone.gif" alt="Crash 684 T-bone" width="90%">
</p>

A reconstructed T-bone crash case.  
**The EA value is the magnitude of the red arrow shown in the visualization. A longer red arrow indicates a higher instantaneous risk level.**

---

## Overview

Evasive Acceleration (EA) quantifies driving risk as the minimum constant relative acceleration required to make a predicted interaction collision-free. Unlike time-to-collision (TTC)-based methods, EA evaluates risk in a two-dimensional joint-motion space rather than through one-dimensional temporal proximity alone.

This repository provides an implementation of EA for two practical use cases:

1. **Single-frame computation**  
   For real-time or instant-evaluation scenarios, such as online risk monitoring, reinforcement learning, and other applications that need the EA value of the current frame immediately.

2. **Batch computation**  
   For offline processing of trajectory datasets, where EA and baseline metrics are computed frame by frame from CSV files.

In addition to EA, the repository also includes a set of baseline risk metrics for comparison, as well as visualization utilities for rendering interaction cases as GIFs.

---

## Why EA?

Most existing risk metrics in autonomous driving are built around the TTC paradigm. These methods quantify risk mainly through a single temporal dimension, even though real traffic interactions are inherently two-dimensional and directional.

This mismatch can lead to two common issues:

- interactions with substantially different avoidance difficulty may receive similar TTC-like values;
- risk may appear to keep increasing even after an effective evasive manoeuvre has already started resolving the conflict.

EA addresses this by measuring the minimum instantaneous intervention required to make the interaction collision-free. It reframes risk as the physical effort needed to restore safety, rather than as time remaining under nominal motion.

---

## Repository structure

```text
evasive-acceleration/
├── README.md
├── LICENSE
├── .gitignore
├── assets/
│   └── gifs/
├── demo_data/
├── src/
│   ├── core_ea.py
│   ├── baseline_risk_metrics.py
│   ├── single_frame.py
│   └── batch_compute.py
└── visualization/
