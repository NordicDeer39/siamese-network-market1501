

# Siamese Network for Person Re-Identification

## Overview

This repository contains the implementation of a Siamese Network for person re-identification using the Market-1501 dataset from Kaggle. The project focuses on leveraging metric learning techniques to compare and match images of pedestrians captured in different camera views.

## Project Details

- **Objective:** To develop a Siamese Network that can effectively re-identify individuals across different camera angles using the Market-1501 dataset.
- **Dataset:** The Market-1501 dataset is a large-scale person re-identification dataset that includes over 32,000 annotated images of 1,501 identities captured by six cameras.
- **Methodology:** The Siamese Network is trained to minimize the distance between embeddings of images of the same person while maximizing the distance for images of different individuals.

## Repository Contents

- `data/`: Instructions for downloading the Market-1501 dataset and organizing it for use in the project.
- `models/`: The architecture and training script for the Siamese Network.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, training, and evaluation of the model.
- `scripts/`: Python scripts for running training and inference on the Siamese Network.
- `results/`: Contains outputs and performance metrics of the model on the Market-1501 dataset.

## Usage

1. **Dataset Setup:**
   - Download the Market-1501 dataset from [Kaggle](https://www.kaggle.com/pengcw1/market-1501) and place it in the `data/` directory as specified.

2. **Preprocessing:**
   - The `notebooks/preprocessing.ipynb` notebook guides you through the preprocessing steps necessary to prepare the dataset for model training.

3. **Training the Model:**
   - Train the Siamese Network using the `notebooks/train_model.ipynb` or by running the training script in the `scripts/` directory.

4. **Evaluating Performance:**
   - Evaluate the model's performance on the Market-1501 dataset using the `notebooks/evaluate_model.ipynb` notebook.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required Python packages using:

```bash
pip install -r requirements.txt
