# Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent credit card transactions using advanced classification techniques and handling imbalanced datasets.

## 📋 Overview

This project demonstrates how to build and tune a machine learning model for fraud detection, addressing common challenges in real-world scenarios such as:

- Handling highly imbalanced datasets
- Evaluating model performance with appropriate metrics
- Fine-tuning models for specific business requirements

## 🎯 Project Objectives

- Preprocess and clean transaction data
- Address class imbalance (0.173% fraud cases) using SMOTE
- Implement and compare classification models:
  - Logistic Regression (baseline)
  - Random Forest (advanced)
- Evaluate models using:
  - Confusion Matrix
  - Precision
  - Recall
- Optimize model threshold for improved fraud detection

## 💾 Dataset

The project uses the "Credit Card Fraud Detection" dataset from Kaggle.

**Dataset Characteristics:**

- **Features:** 30 features (V1-V28 anonymized via PCA)
- **Non-anonymized Features:** Amount and Time
- **Class Distribution:**
  - Total Transactions: 284,807
  - Fraud Cases: 492 (0.173%)
  - Normal Cases: 284,315 (99.827%)

## 🛠️ Methodology

### 1. Data Preprocessing

- Removed Time column (low predictive value)
- Scaled Amount column using StandardScaler

### 2. Handling Class Imbalance

- Split data into train/test (70/30)
- Applied SMOTE to training data only
- Created balanced training set (50/50 split)

### 3. Model Development

- **Baseline Model:** Logistic Regression
  - High Recall but poor Precision
  - 1,967 false positives (not business-viable)
- **Advanced Model:** Random Forest
  - Better balance of Precision (89%) and Recall (78%)
  - Default settings showed promising results

### 4. Model Optimization

- Initial Random Forest (0.5 threshold):
  - 32 missed fraud cases
- Optimized Random Forest (0.3 threshold):
  - Reduced missed frauds to 25
  - Increased false positives from 15 to 45

## 📈 Results

### Model Performance Comparison

| Metric          | Random Forest (Default) | Random Forest (Tuned) |
| --------------- | ----------------------- | --------------------- |
| Threshold       | 0.5                     | 0.3                   |
| Precision       | 89%                     | 82%                   |
| Recall          | 78%                     | 85%                   |
| False Negatives | 32                      | 25                    |
| False Positives | 15                      | 45                    |

## 🔧 Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- imbalanced-learn
- matplotlib
- seaborn
