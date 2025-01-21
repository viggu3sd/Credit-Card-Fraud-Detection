Here is the README file for Credit Card Fraud Detection project:

---

# Credit Card Fraud Detection System

This project implements a **Credit Card Fraud Detection System** using various machine learning models to identify fraudulent transactions. The system leverages data preprocessing, exploratory data analysis (EDA), dimensionality reduction techniques, and model evaluation to achieve optimal performance.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementations](#model-implementations)
- [Evaluation Metrics](#evaluation-metrics)
- [User Interface](#user-interface)
- [Results](#results)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

---

## Overview
The Credit Card Fraud Detection System is designed to classify transactions as fraudulent or non-fraudulent. It emphasizes minimizing **Type-II errors** (False Negatives) to ensure fraudulent transactions are not overlooked.

## Features
- Data preprocessing including handling missing values, normalization, and encoding.
- Dimensionality reduction using PCA and t-SNE.
- Machine learning models:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Linear Discriminant Analysis (LDA)
  - K-Nearest Neighbors (KNN)
- Evaluation of model performance with metrics such as precision, recall, F1-score, AUC, and ROC curve.
- Interactive User Interface for real-time predictions using Gradio.

---

## Technologies Used
- **Python** (Jupyter Notebook)
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Gradio

---

## Data Preprocessing
1. **Data Loading and Inspection**:
   - Loaded dataset to analyze its structure and content.
2. **Data Cleaning**:
   - Handled missing values and removed unnecessary columns like `customerID`.
3. **Normalization**:
   - Applied z-score normalization to numerical data.
4. **Encoding**:
   - Label Encoding for ordinal categorical data.
   - One-Hot Encoding for nominal categorical data.

---

## Model Implementations
1. **Logistic Regression**:
   - Basic and polynomial feature augmentation.
   - Emphasis on recall for fraud detection.
2. **Support Vector Machines (SVM)**:
   - Linear, Polynomial, and RBF kernels with hyperparameter tuning.
3. **Linear Discriminant Analysis (LDA)**:
   - Linear classification approach for feature separation.
4. **K-Nearest Neighbors (KNN)**:
   - Optimal neighbor selection via cross-validation.

---

## Evaluation Metrics
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Relevance of predicted positive cases.
- **Recall**: Ability to capture true positives.
- **F1-Score**: Balance between precision and recall.
- **AUC**: Area under the ROC curve to summarize performance.

---

## User Interface
- An interactive UI is developed using **Gradio**.
- Allows users to input transaction details and receive fraud probability predictions in real-time.

---

## Results
1. Logistic Regression:
   - Accuracy: 78.7%, AUC: 82.6%, Recall: 56.5%
2. SVM:
   - Best Kernel: RBF
   - AUC: 80.2%, Recall: 42.6%
3. KNN:
   - Optimal Neighbors: 20
   - Accuracy: 77%, AUC: 79.6%
4. LDA:
   - Accuracy: 77.1%, AUC: 81.4%, Recall: 53.5%

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory and open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook for model training and evaluation.
5. Launch the Gradio interface:
   ```bash
   python app.py
   ```

---

## Acknowledgments
- This project was developed as part of an **Introduction to Machine Learning** course.
- Special thanks to the dataset providers and open-source community.

---
