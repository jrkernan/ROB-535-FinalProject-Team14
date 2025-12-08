# RMT-PPAD — Multi-Task Drivable-Area Segmentation Evaluation

This directory contains the evaluation setup used in our project.

RMT-PPAD is a multi-task perception framework designed for autonomous driving that jointly predicts drivable area, lane markings, and other road structure cues. In our study, we evaluate the authors’ **pretrained RMT-PPAD model** on the **BDD100K validation set** to assess how multitask learning impacts drivable-area segmentation performance.

---

## Original Repository

All model architecture details, training procedures, dataset conventions, and environment setup instructions are described in the original repository:

> **RMT-PPAD GitHub:**  
> https://github.com/JiayuanWang-JW/RMT-PPAD

Users should follow the original repository to:
- install dependencies
- configure the Python environment
- download pretrained checkpoints
- prepare the BDD100K dataset

---

## Dataset and Configuration Setup

RMT-PPAD relies on several configuration files that contain hardcoded file paths.
These configuration files are referenced in the original repository and **must be updated** to match your local directory structure before running evaluation.

---

## Evaluation Script Used in This Project

The original repository provides evaluation utilities, but they do not directly expose all metrics required for our analysis (e.g., **F1 score for drivable-area segmentation**).

To address this, we use a custom evaluation script:

```text
detailed_results.py
