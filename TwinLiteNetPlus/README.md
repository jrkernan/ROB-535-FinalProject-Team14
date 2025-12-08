# TwinLiteNet+ — Drivable-Area Segmentation Evaluation

This directory contains the evaluation setup used in our project.

TwinLiteNet+ is a lightweight neural network architecture specialized for drivable-area prediction. In our study, we evaluate the **pretrained TwinLiteNet+ model** on the **BDD100K validation set** using the authors’ original evaluation pipeline.

---

## Original Repository

All model architecture details, training procedures, and dependency information are provided by the original authors:

> **TwinLiteNet+ GitHub:**  
> https://github.com/chequanghuy/TwinLiteNetPlus

Please follow the instructions in the original repository to:
- install dependencies
- set up the environment
- download the pretrained model
- download and organize the BDD100K dataset

---

## Model Variant Used

TwinLiteNet+ offers multiple model configurations.  
**To reproduce our reported results, you must use the *Large* model variant provided by the authors.**

---

## Reproducing Our Evaluation Results

Once dependencies are installed and the dataset is correctly structured, reproduce our results by running:

```bash
python val.py --config 'large' --weight 'pretrained/large.pth'
