# Benchmarking State-of-the-Art Deep Learning Paradigms for Drivable-Area Segmentation

This repository contains the code and evaluation artifacts for a comparative study of modern deep learning paradigms applied to **drivable-area segmentation**, a core perception task for self-driving systems.

We benchmark how a model’s training scope and task responsibility impact performance, contrasting:
- task-specialized architectures,
- multi-task perception frameworks designed for self-driving,
- and general-purpose transformer models with global scene reasoning.

Our goal is to evaluate whether increasing contextual breadth improves drivable-area segmentation, or if narrow task specialization yields superior accuracy and efficiency for safety-critical perception.

![Image](https://github.com/user-attachments/assets/576b51b4-bc9d-41eb-ba56-ceb4afdde6ac)
![Image](https://github.com/user-attachments/assets/ffbfdc3c-7e44-405f-8073-4945cf012310)


## Models Evaluated

We compare three representative paradigms:

- **Mask2Former (2023)**  
  A general-purpose transformer-based semantic segmentation model trained on broad visual domains, not tailored specifically for autonomous driving.

- **RMT-PPAD (2025)**  
  A multi-task perception framework designed for self-driving that jointly infers drivable area, lane structure, and traffic agents.

- **TwinLiteNet+ (2025)**  
  A lightweight architecture specialized explicitly for drivable-area segmentation.

---

## Evaluation Overview

All models were evaluated on the **BDD100K** dataset using its binary drivable-area annotations.  
Pretrained models were used for all experiments. Due to architectural differences, each model required a distinct evaluation workflow, implemented within its corresponding subdirectory.

---

## Quantitative Results

| Model         | Pixel Accuracy | IoU   | F1 Score | FLOPs  |
|--------------|----------------|-------|----------|--------|
| Mask2Former  | 0.882          | 0.543 | 0.704    | 792.5G |
| RMT-PPAD     | 0.942          | 0.875 | 0.931    | 209.3G |
| TwinLiteNet+ | 0.979      | **0.884** | 0.938 | **17.58G** |

---

## Key Findings

- General-purpose transformer models leverage global context for scene understanding but struggle in complex road layouts such as multi-lane or multi-directional traffic.
- Reducing task scope and model responsibility improves drivable-area boundary precision while significantly lowering computational cost.
- For safety-critical perception, task-specialized or narrowly scoped models offer the strongest accuracy–efficiency trade-off, suggesting modular perception systems may be preferable to fully unified architectures.

---

## Reproducibility

Detailed instructions to reproduce results are provided within each model subdirectory, including setup, inference, and evaluation steps
