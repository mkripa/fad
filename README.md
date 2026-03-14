# Explainable Multimodal LLMs for Audio Deepfake Detection

This repository provides an end-to-end, reproducible pipeline for supervised fine-tuning and explainability of multimodal large language models (MLLMs) for audio deepfake detection.

The implementation serves as a template for finetuning multimodal LLMs to classification tasks and analyzing their decision behavior using surrogate modeling and XAI techniques.

---

## Dataset

Experiments use ASVspoof 2019 (LA) dataset. The dataset can be downloaded from the official ASVspoof website: https://www.asvspoof.org/index2019.html

After downloading the dataset, convert the audio files to .wav format and place them in the corresponding real/ and fake/ folders for fine-tuning.

---

## Supervised Fine-tuning

_sft/sft_gemma.ipynb_ demonstrates LoRA-based SFT pipeline for Gemma-3N using ASVspoof dataset.

For Qwen, the fine-tuning and experimental setup follow the approach described in: https://arxiv.org/pdf/2505.11079 

---
## Installation

```bash
pip install -r requirements.txt
```

## Full pipeline
```bash
python run.py --config config.yaml
```

## Run specific steps
```bash
python run.py --config config.yaml --steps 4 5 
```
---

## Problem Setting

Given an audio sample, the model must classify it as:

- **Real**
- **Fake**

Unlike traditional acoustic classifiers, this pipeline fine-tunes a multimodal LLM and then explains its predictions using standard acoustic features and model-agnostic interpretability methods.

---

## Pipeline Overview

The system implements a four-stage modular workflow:

### 1) Supervised Fine-Tuning (SFT)

- _sft/sft_gemma.ipynb_
  
### 2) Model Evaluation & Prediction Export

- _evaluate.py_

### 3) Acoustic Feature Extraction (openSMILE)

- Feature set: **eGeMAPSv02**
- 88 acoustic descriptors per audio sample (tabular representation)

### 4) Surrogate Modeling + Explainability (XAI Layer)

Surrogate model:
- Random Forest

Explainability methods:
- **SHAP** (TreeExplainer): global feature attribution across the dataset
- **LIME** (LimeTabularExplainer): local explanations for individual instances

---

The pipeline is model-agnostic and can be applied to any multimodal LLM that supports audio input.
### Demonstrated Models


| Model    | Fine-Tuned in This Repo      |  Explainability Applied |
|----------|------------------------------|-------------------------|
| Gemma-3N | ✅                           | ✅                      |
| Qwen     | (obtained from prior work)   | ✅                      |


The same surrogate + XAI workflow is applied to both models to enable comparative interpretability analysis.

---

