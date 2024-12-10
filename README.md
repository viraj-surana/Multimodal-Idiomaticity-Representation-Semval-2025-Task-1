# Multimodal-Idiomaticity-Representation-Semval2025

# Project By
Viraj (22BDS065)

## Overview
This repository contains the implementation for **SemEval-2025 Task 1: AdMIRe (Advancing Multimodal Idiomaticity Representation)**. The task focuses on using multimodal data (text and images) to improve idiomaticity understanding and representation, with two subtasks:

1. **Subtask A:** Identify which image best represents an idiomatic expression in a given sentence.
---
2. **Subtask B:** Identify where can we place the generated image based on the order of how that best represents an idiomatic expression in a given sentence.
---

## Introduction
Idioms are multi-word expressions with meanings often unrelated to the literal meanings of their components, posing challenges for NLP systems. This task builds on SemEval-2022 Task 2, advancing idiomaticity representation using multimodal data.

Effective idiom understanding is critical for tasks like:
- Sentiment analysis
- Machine translation
- Natural language understanding

---

## Methodology
Our approach integrates textual and visual features to rank images based on their relevance to idiomatic expressions in context.

Key steps include:
1. **Data Preprocessing**:
   - Tokenize text using BERT tokenizer.
   - Normalize and resize images to match ResNet-50 input requirements.
2. **Feature Extraction**:
   - Textual features from **BERT** embeddings (768-dimensional).
   - Visual features from **ResNet-50** (2048-dimensional).
3. **Feature Fusion**:
   - Combined textual and visual embeddings into a 128-dimensional space.
4. **Ranking Prediction**:
   - A neural network predicts rankings for each image.

---

## Dataset
The dataset consists of:
- Context sentences containing idiomatic expressions.
- Associated images depicting literal and idiomatic meanings.

Data processing details:
- Text sequences padded/truncated to 128 tokens.
- Images resized to 224x224 pixels and normalized.

---

## Model Architecture
1. **Text Encoder**: 
   - Uses **BERT** for sentence embeddings.
   - Outputs dense 128-dimensional vectors.

2. **Image Encoder**: 
   - Uses **ResNet-50**, pre-trained on ImageNet.
   - Outputs 128-dimensional image embeddings.

3. **Fusion Layer**:
   - Combines text and image embeddings for ranking.

4. **Ranking Layer**:
   - Fully connected neural network outputs scores.

---

## Training and Evaluation
1. **Loss Function**:
   - Mean Squared Error (MSE) to optimize ranking accuracy.

2. **Optimizer**:
   - Adam optimizer with learning rate tuning for efficient convergence.

3. **Evaluation Metric**:
   - Mean Reciprocal Rank (MRR) for ranking effectiveness.

### Results
| Model | MRR Score |
|-------|-----------|
| Model 1 | 0.2940 |
| Model 2 | 0.5167 |
| Model 3 | 0.5321 |

---

## How to Run the Code
### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- OpenCV
- NumPy and Pandas

