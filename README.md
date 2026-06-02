# Tree Branch & Needle Tip Extraction Pipeline

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-computer--vision-green)
![NetworkX](https://img.shields.io/badge/graph-NetworkX-red)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

A computer vision pipeline for detecting a needle tip and extracting connected branch structures using skeletonisation and graph analysis.

---

## Overview

The pipeline processes images to:

- Detect the needle tip
- Extract branch structure
- Build a skeleton-based graph
- Find the longest branch path from the tip

Two modes are supported:
- Single image processing
- Background subtraction mode

---

## Pipeline

### 1. Input Selection
- Load single images or background + target pair
- Optional background subtraction

---

### 2. Tip Detection
- Segment needle region
- Skeletonise structure
- Use PCA + endpoints to locate tip

---

### 3. Branch Extraction
- Enhance branch-like structures (vesselness filter)
- Threshold and clean image
- Skeletonise branches

---

### 4. Graph Analysis
- Convert skeleton to graph
- Extract connected structure from tip
- Find longest path along branches

---

### 5. Visualisation
- Overlay skeleton, tip, and main branch path on original image
- Optional debug display at each step
  
