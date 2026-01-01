

 – Intrusion Detection System (UNSW‑NB15 + Improved XGBoost Model)



1. Dataset Information

Dataset Used: UNSW‑NB15 (Train & Test CSV files)
Download Link:
https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Files required in the project folder:

.UNSW_NB15_training-set.csv

.UNSW_NB15_testing-set.csv


2. How to Run the Project

1.First of all create virtual environement

install python3.10 in your system

Then create folder and open in open VS Code

Run commands in Terminal in sequence:

. python3.10 --version

. python3.10 -m venv myenv 

. .\myenv\Scripts\activate



2.Install Required Libraries in terminal

pip install numpy pandas scipy scikit-learn xgboost lightgbm joblib matplotlib seaborn imbalanced-learn

3.Run the Training Script to check the model perfomance
python app2binary.py

4.Run Frontned 
. streamlit run FrontendStreamlit.py

3. Output Files (Artifacts)
After running the script, the folder artifacts_improved/ will contain automatically:

. attack_distribution.png
. classification_report.txt
. confusion_matrix.png
. label_encoder.joblib
. lgb_binary_model.txt
. preprocessor_sparse.joblib
. train_medians.joblib


4. Model Architecture Summary
.Sparse‑aware preprocessing pipeline
.One‑Hot‑Encoded categorical features
.Standardized numerical features
.Inverse‑frequency class balancing
.Optional GBAC iterative weighting
.Multi‑class XGBoost (multi:softprob)
.Early stopping for best model selection

5. Evaluation Metrics (Your Model Results)
.Metric	Value
.Accuracy	90.05%
.Macro F1‑Score	86.26%
.Confusion Matrix	Saved as image + CSV
.Detailed classification metrics are printed automatically after training.

6. Project Structure
📁 IntrusionDetectionSystem/
|── artifacts_improved/
│── app2binary.py
│── FrontendStreamlit.py
│── README.md
│── requirements.txt
│── UNSW_NB15_training-set.csv
│── UNSW_NB15_testing-set.csv

