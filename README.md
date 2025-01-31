# Maternal Health Risk Analysis with PySpark

## 📌 Overview
This repository contains an analysis of the **Maternal Health Risk Dataset** using **PySpark**. The project explores different machine learning models to predict maternal health risk levels, including **Logistic Regression, Random Forest, XGBoost, and Artificial Neural Networks (ANNs)**. Additionally, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to handle class imbalance.

## 📂 Repository Structure 
│── 📄 **README.md**  # Project Documentation  
│── 📄 **main_notebook.ipynb**  # Main analysis notebook (PySpark models & evaluation)  
│── 📄 **smote_script.py**  # SMOTE resampling script  
│── 📄 **smote_resampled_dataset.csv**  # Dataset after SMOTE resampling


## 📊 Dataset
The dataset used for this analysis is available on Kaggle:  
🔗 **[Maternal Health Risk Dataset](https://www.kaggle.com/code/anuz505/maternal-health-risk-analysis-with-pyspark)**

### **Features:**
- **Age:** Age of the patient
- **SystolicBP:** Systolic Blood Pressure
- **DiastolicBP:** Diastolic Blood Pressure
- **BS:** Blood Sugar level
- **BodyTemp:** Body Temperature
- **HeartRate:** Heart Rate
- **RiskLevel:** Low, Medium, or High risk category

## 🏗️ Models Implemented
1. **Logistic Regression** – Baseline model (61% accuracy)
2. **Random Forest Classifier** – Best performing model (74% accuracy)
4. **Artificial Neural Network (PySpark MLP)** – Custom ANN with a multi-layer structure
5. **SMOTE Resampling** – Applied to improve class balance before training models

## ⚙️ How to Run
### **1️⃣ Running the Main Notebook**
Run the `main_notebook.ipynb` to execute all models and compare results.
```bash
jupyter notebook main_notebook.ipynb 
```

# 2️⃣ SMOTE Resampling

To generate a resampled dataset with SMOTE, run:

```bash
python smote_script.py
```
## 🚀 Results & Key Findings
--**Logistic Regrssion** average performance since it is not suited for the datset.
- **Random Forest** outperformed all other models due to better handling of overlapping features.
- **ANN (PySpark MLP)** was experimented with, but hyperparameter tuning is needed for better results.
- **SMOTE resampling** helped improve class balance, but further tuning is required for better generalization.

---

## 🔗 Kaggle Notebook

- [📌 Kaggle Analysis Notebook: Maternal Health Risk Analysis with PySpark](#)

---



