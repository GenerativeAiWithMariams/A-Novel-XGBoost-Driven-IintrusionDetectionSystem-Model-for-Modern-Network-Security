

# 🔐 Intrusion Detection System (IDS) Predictor

This project demonstrates an **Intrusion Detection System Predictor** using **LightGBM**, **scikit-learn preprocessing**, and **Streamlit** for an interactive web interface.
It allows both **single instance** and **batch CSV predictions** for detecting network attacks.

---

## 📌 Project Concept

This project is based on **supervised machine learning** for binary classification:

* **Goal**: Predict whether a network record is an **Attack** or **Normal**.
* **Input**: Network features (duration, bytes, packets, protocol, etc.)
* **Output**: Probability of **Attack** and predicted label

### 🤖 Predictor Behavior:

* Accepts either **manual input** (key=value pairs) or **CSV batch upload**
* Handles **feature preprocessing** automatically
* Fills missing numeric values using **training medians** (if available)
* Generates predictions using **LightGBM**
* Visualizes results (bar charts, distribution plots)
* Allows downloading **prediction results CSV**

---

## ⚙️ Technologies Used

* **Python 3.10+**
* **Streamlit** – interactive web interface
* **LightGBM** – machine learning model
* **Joblib** – load pre-trained artifacts (preprocessor, model, label encoder)
* **Pandas / NumPy** – data handling
* **SciPy sparse** – handle sparse matrices from preprocessor
* **Matplotlib & Seaborn** – visualization

---

## 📁 Files

```
📁 IntrusionDetectionSystem/
│── artifacts_improved/                 # Contains pre-trained model artifacts
│   ├── preprocessor_sparse.joblib
│   ├── label_encoder.joblib
│   ├── lgb_binary_model.txt
│   └── train_medians.joblib
│
│── app2binary.py                       # Python script for single/batch predictions
│── FrontendStreamlit.py                # Streamlit UI for IDS prediction
│── README.md                           # Project documentation
│── requirements.txt                    # Python dependencies
│── UNSW_NB15_training-set.csv          # Training dataset (optional for reference)
│── UNSW_NB15_testing-set.csv           # Testing dataset (optional for reference)

```

---

## 📦 Installation

### 1️⃣ Install Required Libraries

```bash
pip install streamlit numpy pandas scipy joblib lightgbm matplotlib seaborn
```

---

## ▶️ How to Run the Project

```bash
streamlit run FrontendStreamlit.py
```

A browser window will open showing the interactive IDS prediction interface.

---

## 🧠 Key Components Explained

### parse_input_block(text)

* Parses `key=value` pairs for single-row input
* Applies **alias mapping** (e.g., `duration` → `dur`)
* Returns a dictionary ready for preprocessing

### coerce_value(v)

* Converts string inputs to numeric values where possible
* Handles missing values as `NaN`

### dense_from_proc(X_proc)

* Converts sparse preprocessed matrix to dense array for display and prediction

### load_artifacts(preproc, labelenc, model, medians)

* Loads required artifacts:

  * Preprocessor pipeline
  * Label encoder
  * LightGBM model
  * Optional training medians for numeric feature imputation

### Prediction Flow

**Single Prediction Mode**:

1. User enters input in `key=value` format
2. Preprocessor transforms input features
3. LightGBM predicts probability for **Attack**
4. Threshold slider determines final label
5. Displays processed vector, predicted probability, and a small bar chart

**Batch CSV Prediction Mode**:

1. Upload CSV with network records
2. Missing columns automatically added as `NaN`
3. Numeric missing values filled from training medians
4. Preprocessed and predicted in batch
5. Downloadable CSV output
6. Prediction distribution plot visualized

---

## 📊 Application Output

* Real-time **prediction feedback**
* **Processed feature vectors** preview
* **Probability of Attack** and predicted label
* **Batch prediction** with downloadable CSV
* **Distribution charts** for multiple predictions

---

## 🎯 Learning Outcomes

* Handling **sparse and dense feature matrices** for ML models
* Integrating **LightGBM** with Streamlit
* Parsing flexible user input formats
* Filling missing values using **training medians**
* Visualizing predictions and distributions
* Building **robust single and batch prediction pipelines**

---

## 🚀 Future Improvements

* Extend with **SHAP explainability** for feature attribution
* Add **real-time network traffic ingestion**
* Integrate **multi-class attack detection**
* Add **user authentication** for secure deployment
* Automate **artifact update & versioning**

---

## 👩‍💻 Author

MaryamS
GenAI Developer | AI | Machine Learning | Python Enthusiast

---

✅ Application Complete — IDS Predictor ready to classify network records with probability-based attack detection!


