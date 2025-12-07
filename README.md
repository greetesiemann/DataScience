# Breast Cancer Outcome Prediction Using Machine Learning
Group D8 — Triin-Elis Kuum, Greete Siemann, Hanna Samelselg

## Project Motivation
Breast cancer is one of the most common cancers worldwide, and early diagnosis greatly improves treatment outcomes.
The goal of this project is not to create a clinical diagnostic tool, but to:
- understand how machine-learning models behave on biomedical data
- compare multiple classification algorithms
- practice the CRISP-DM methodology
- build a fully reproducible data science workflow
We use the Wisconsin Diagnostic Breast Cancer dataset, which includes 569 samples and 30 numerical features describing tumor cell characteristics. Each sample is labeled as benign or malignant.

## Project Goal
Our main objective is to develop and evaluate several machine-learning models for predicting whether a tumor is benign (0) or malignant (1).
Models included:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (Linear kernel)
- Support Vector Machine (RBF kernel)
We evaluate models using accuracy, precision, recall, F1-score and ROC–AUC, then choose the best-performing classifier.

## Notebook Sections
Section	Description
Task 1 — Setup – Repository creation & environment preparation
Task 2 — Business Understanding – Goals, success criteria, risks
Task 3 — Data Understanding – Data inspection, cleaning, EDA
Task 4 — Data Preparation – Train/test split, scaling, encoding
Modeling – Train 4 different ML models
Evaluation – Metrics, confusion matrices, ROC curve
Feature Analysis – Feature importance (Random Forest)
Results & Conclusions – Comparison and final discussion
The notebook is fully executable from top to bottom and contains comments explaining each step.

## How to Reproduce the Analysis
Follow these steps to run the notebook exactly as the authors did.
### 1. Install Requirements
You need Python 3.8+. The project uses:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install all dependencies with:
- pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:
- pip install pandas numpy matplotlib seaborn scikit-learn

### 2. Open the Jupyter Notebook
- jupyter notebook

Then open:
- DataScienceProject.ipynb

### 3. Run Notebook Cells in Order
Simply run all cells from top to bottom.
The notebook will automatically:
- load and clean the dataset
- run exploratory visualizations
- preprocess data (scaling + encoding)
- train 4 machine-learning models
- produce metrics
- generate ROC curve
- display confusion matrices
- output tables comparing all models

### 4. Expected Final Output
You will see:
- all classification reports
- 4 confusion matrices
- ROC curve for best model
- feature importance bar plot
- summary comparison table
If everything runs correctly, the SVM RBF should achieve AUC ≈ 0.99 and accuracy around 0.95–0.98 depending on random split.

## Notes for Reviewers
- The repository contains all code, including parts not shown in the poster.
- Code is written to be readable and includes comments explaining logic.
- All experiments are reproducible — we fix random_state=42 whenever relevant.
- No patient-identifying information is used; dataset is public and safe.

## Authors
Group D8:
- Triin-Elis Kuum
- Greete Siemann
- Hanna Samelselg
