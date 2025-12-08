


 – Intrusion Detection System (UNSW‑NB15 + Improved XGBoost Model)



1. Project Overview
This project implements a multi‑class Intrusion Detection System (IDS) using the UNSW‑NB15 dataset.

A novel, optimized pipeline was created based on:

Advanced data preprocessing

Sparse matrix transformation

Class imbalance handling

Improved multi‑class XGBoost model

Detailed evaluation & visualization

The output model detects 10 attack categories 
including DoS, Reconnaissance, Exploits, Fuzzers, Shellcode, Worms, Generic, Backdoor, Analysis, and Normal traffic.


2. Dataset Information

Dataset Used: UNSW‑NB15 (Train & Test CSV files)

Download Link:

https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Files required in the project folder:

.UNSW_NB15_training-set.csv
.UNSW_NB15_testing-set.csv


3. How to Run the Project

1.Install Required Libraries in terminal

pip install numpy pandas scipy scikit-learn xgboost lightgbm joblib matplotlib seaborn imbalanced-learn

2️.Run the Training Script

python app.py


4. Output Files (Artifacts)
After running the script, the folder artifacts_improved/ will contain automatically:

File Purpose

.preprocessor_improved.joblib	Full preprocessing pipeline

.label_encoder_improved.joblib	Encodes attack category labels

.xgb_multiclass_improved.model	Final trained IDS model

.confusion_matrix.png	Heatmap of class predictions

.confusion_matrix.csv	Numerical confusion matrix


5. Model Architecture Summary
   
.Sparse‑aware preprocessing pipeline

.One‑Hot‑Encoded categorical features

.Standardized numerical features

.Inverse‑frequency class balancing

.Optional GBAC iterative weighting

.Multi‑class XGBoost (multi:softprob)

.Early stopping for best model selection


7. Evaluation Metrics (Your Model Results)
   
.Metric	Value

.Accuracy	83.52%

.Macro F1‑Score	0.6180

.Confusion Matrix	Saved as image + CSV

.Detailed classification metrics are printed automatically after training.


9. Project Structure

📁 IntrusionDetectionSystem/
|── artifacts_improved/
│── app.py
│── UNSW_NB15_training-set.csv
│── UNSW_NB15_testing-set.csv
│── README.md
│


