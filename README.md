# Sentiment Analysis Model for News Articles on Sanctions Against Russia(https://www.overleaf.com/project/663218b792e8d2f6e079e18c)

## Overview
This repository contains the code and documentation for a sentiment analysis model tailored specifically for news articles discussing sanctions against Russia. The model utilizes advanced natural language processing techniques and is built on top of the DistilBERT architecture, optimized for efficient computation and high accuracy.

## Model Architecture
The core of the model architecture consists of a DistilBERT base model with a classification head added on top. Additionally, the model incorporates Low-Rank Adaptation (LoRA) techniques to reduce computational complexity while preserving performance.

### Components:
- **DistilBERT Base Model**: Pre-trained on the `distilbert-base-multilingual-cased` checkpoint.
- **Classification Head**: A feed-forward neural network for the sentiment classification task.
- **LoRA (Low-Rank Adaptation)**: Fine-tuning technique for efficient adaptation of pre-trained models.

## Training
The training process involves meticulous hyperparameter tuning and optimization to achieve the best possible performance.

### Hyperparameter Tuning:
Hyperparameters are tuned using the Optuna framework to maximize validation accuracy. Tuned hyperparameters include learning rate, batch size, number of epochs, and LoRA parameters.

### Training Procedure:
- Data preprocessing includes tokenization, padding, and numerical conversion.
- Multiple training trials are conducted with varying hyperparameters.
- Training progress is tracked using Weights and Biases (W&B).
- Validation metrics monitored include loss, accuracy, F1 score, precision, and recall.

## Model Performance
The best-performing model achieved remarkable results on both training and validation datasets.

### Training and Validation Results:
- Detailed performance metrics are provided for each epoch in the training process.
- The best model's hyperparameters are listed for reference.

## Optimization Results
Hyperparameter optimization results are visualized for better understanding and analysis.

## Conclusion
The developed sentiment analysis model offers valuable insights into the sentiment landscape surrounding sanctions against Russia. Its robust performance and efficient architecture make it a useful tool for policymakers, analysts, and the public to gauge the impact of such measures.
