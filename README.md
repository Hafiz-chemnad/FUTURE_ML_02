# AI-Powered Churn Prediction System - Future Interns ML Internship Task 2

## Project Overview
This repository contains the solution for Task 2 of the Future Interns Machine Learning Internship. The project focuses on building an AI-powered system to predict customer churn, a critical business problem aimed at improving customer retention strategies. It covers the entire machine learning pipeline from data understanding to model deployment (conceptual within the notebook) and actionable insights.

## Objective
The primary objective of this task was to develop a robust machine learning model capable of predicting which customers are most likely to churn. This involves identifying key factors contributing to churn, assessing model performance using various metrics, and translating findings into practical business recommendations for customer retention.

## Dataset
The project utilized the Telco Customer Churn dataset, which provides comprehensive information about a telecommunications company's customers. This dataset includes demographic details, services subscribed to, account information, and most importantly, a binary target variable indicating whether a customer has churned.

* **Source:** [(https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)]


## Methodology
The project followed a systematic approach to develop the churn prediction system:

### 1. Data Loading & Initial Inspection
Loaded the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset and performed initial checks on its structure, data types, and missing values. Identified the presence of class imbalance in the 'Churn' target variable and the 'TotalCharges' column needing type conversion.

### 2. Exploratory Data Analysis (EDA)
Conducted in-depth analysis to uncover patterns and relationships between customer attributes and churn. Visualizations were used to highlight key churn drivers such as contract type, internet service, additional services, payment method, tenure, and monthly charges.

### 3. Data Preprocessing & Feature Engineering

#### Handling `TotalCharges` Column:
   * Handled the `TotalCharges` column by converting it to numeric and imputing missing values with `0`.

#### Removing `customerID`:
   * Removed the `customerID` as it's not a predictive feature.

#### Transforming Categorical Features:
   * Categorical features were transformed using One-Hot Encoding.

#### Scaling Numerical Features:
   * Numerical features were scaled using StandardScaler.

#### Addressing Class Imbalance:
   * Addressed class imbalance by applying **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data to balance the 'Churn' classes.

### 4. Model Building & Training
Selected and trained three classification models: Logistic Regression, Random Forest Classifier, and XGBoost Classifier. Models were trained on the preprocessed and balanced training dataset.

### 5. Model Evaluation
Evaluated model performance on the unseen test set using a comprehensive suite of metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Confusion matrices and ROC curves were plotted for visual assessment. Feature importance was extracted from the XGBoost model to identify the strongest predictors of churn.

## Deliverables
The core deliverable for this task is the Google Colab / Jupyter Notebook (`FUTURE_ML_02.ipynb`) which serves as the predictive system and insights dashboard. It includes:
* All Python code for data loading, preprocessing, EDA, model training, and evaluation.
* Clear and annotated plots showing feature distributions, relationships with churn, confusion matrices, and ROC curves.
* Detailed Markdown cells providing explanations for each step, observations from EDA, model choices, evaluation results, and derived business insights.

## How to Run
To run this notebook and explore the Churn Prediction System:

### 1. Clone this repository: (Optional, if you prefer direct upload from Colab, you can skip this)
   ```bash
   git clone [https://github.com/YourGitHubUsername/FUTURE_ML_02.git](https://github.com/YourGitHubUsername/FUTURE_ML_02.git)
   cd FUTURE_ML_02
