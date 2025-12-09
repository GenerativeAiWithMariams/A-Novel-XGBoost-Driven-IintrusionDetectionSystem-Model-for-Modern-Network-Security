# preprocess_and_train_binary_fixed_v2.py
# Usage: python preprocess_and_train_binary_fixed_v2.py
# Improvements:
#  - OneHotEncoder compatibility (sparse_output fallback)
#  - Automatic leakage detection: drop features that perfectly map to label
#  - Quick train/test overlap check and informative logging
#  - Save training medians (for inference), classification report, confusion matrix and plots

import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import lightgbm as lgb
from collections import Counter

# Paths - adjust if needed
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV  = "UNSW_NB15_testing-set.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=== IDS Binary Training (fixed v2) ===")

# 1) Load CSVs (low_memory)
print("Loading CSVs...")
df_train = pd.read_csv(TRAIN_CSV, low_memory=True)
df_test  = pd.read_csv(TEST_CSV,  low_memory=True)
print("Train rows:", len(df_train), "Test rows:", len(df_test))

# 2) Identify label column
if 'attack_cat' in df_train.columns:
    label_col = 'attack_cat'
elif 'label' in df_train.columns:
    label_col = 'label'
else:
    raise RuntimeError("Label column not found. Edit script with the exact label column name.")
print("Using label column:", label_col)

# 3) Create binary target: Normal vs Attack
def make_binary_label(s):
    s = s.astype(str)
    return np.where(s == 'Normal', 'Normal', 'Attack')

df_train['binary_label'] = make_binary_label(df_train[label_col])
df_test['binary_label']  = make_binary_label(df_test[label_col])
print("Binary labels added.")

# 4) Drop identifier columns if present
drop_cols = [c for c in ['id','ID','flow_id','timestamp','start_time','start','src_ip','dst_ip','sport','dport'] if c in df_train.columns]
if drop_cols:
    print("Dropping identifier columns if present:", drop_cols)
df_train = df_train.drop(columns=drop_cols, errors='ignore')
df_test  = df_test.drop(columns=drop_cols, errors='ignore')

# Quick train/test overlap check (using a small signature of columns)
sig_cols = [c for c in df_train.columns if c not in [label_col, 'binary_label']][:10]
if len(sig_cols) >= 3:
    merged = pd.merge(df_train[sig_cols].drop_duplicates(), df_test[sig_cols].drop_duplicates(), how='inner', on=sig_cols)
    if len(merged) > 0:
        print("WARNING: Detected", len(merged), "rows in common between train and test (based on first signature columns).")
    else:
        print("No obvious train/test overlap detected (signature columns).")
else:
    print("Not enough columns to run train/test overlap signature check.")

# 5) Keep features present in both train & test (and exclude label columns)
feature_cols = [c for c in df_train.columns if c in df_test.columns and c not in [label_col, 'binary_label']]
print("Initial feature columns count:", len(feature_cols))

# 6) Leakage detection: drop any feature that perfectly maps to binary label
def detect_leaky_features(df, label_series, features):
    leaky = []
    for col in features:
        try:
            vals = df[col].fillna('_NA_').astype(str)
            # For each unique feature value, check set of labels observed
            mapping = {}
            for idx, v in vals.items():
                mapping.setdefault(v, set()).add(label_series.loc[idx])
            # If every unique value maps to only one label (no mixing), suspicious
            all_single_label = all(len(s) == 1 for s in mapping.values())
            if all_single_label and len(mapping) > 1:
                leaky.append(col)
        except Exception:
            continue
    return leaky

print("Checking for leaking features (features that perfectly map to the label)...")
leaky_feats = detect_leaky_features(df_train, df_train['binary_label'], feature_cols)
if leaky_feats:
    print("WARNING: Found potentially leaky features that will be removed:", leaky_feats)
    feature_cols = [c for c in feature_cols if c not in leaky_feats]
else:
    print("No perfect-leakage features found.")

print("Final feature cols count after leakage removal:", len(feature_cols))

# 7) Prepare X/y
X_train = df_train[feature_cols].copy()
y_train = df_train['binary_label'].copy()
X_test  = df_test[feature_cols].copy()
y_test  = df_test['binary_label'].copy()

# Save numeric medians for inference fill (useful for Streamlit app)
train_medians = X_train.median(numeric_only=True)
joblib.dump(train_medians, os.path.join(ARTIFACT_DIR, 'train_medians.joblib'))
print("Saved training medians to artifacts for inference filling.")

# 8) Identify numeric & categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# 9) Build memory-efficient preprocessor
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

# compatibility-safe OneHotEncoder creation
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', ohe)])

preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)], remainder='drop', sparse_threshold=0.0)

# 10) Fit preprocessor and transform to sparse CSR
print("Fitting preprocessor (memory-efficient)...")
preprocessor.fit(X_train)

def transform_to_csr(preproc, X, num_cols, cat_cols):
    X_num = preproc.named_transformers_['num'].transform(X[num_cols]) if len(num_cols) > 0 else None
    X_cat = None
    if len(cat_cols) > 0:
        cat_imputer = preproc.named_transformers_['cat'].named_steps['imputer']
        cat_ohe = preproc.named_transformers_['cat'].named_steps['ohe']
        X_cat_imputed = cat_imputer.transform(X[cat_cols])
        X_cat = cat_ohe.transform(X_cat_imputed)
    if X_num is None:
        return X_cat.tocsr()
    if X_cat is None:
        return sparse.csr_matrix(X_num)
    return sparse.hstack([sparse.csr_matrix(X_num), X_cat]).tocsr()

X_train_proc = transform_to_csr(preprocessor, X_train, num_cols, cat_cols)
X_test_proc  = transform_to_csr(preprocessor, X_test, num_cols, cat_cols)
print("Processed shapes (train/test):", X_train_proc.shape, X_test_proc.shape)

# 11) Label encode target
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.astype(str))
y_test_enc  = le.transform(y_test.astype(str))
print("Classes:", le.classes_)

# Safety check: ensure train/test labels look sensible
print("Train label distribution:", dict(Counter(y_train_enc)))
print("Test label distribution :", dict(Counter(y_test_enc)))

# 12) Train LightGBM
print("Training LightGBM...")
train_data = lgb.Dataset(X_train_proc, label=y_train_enc)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 64,
    'max_depth': -1,
    'verbose': -1,
    'seed': 42
}
bst = lgb.train(params, train_data, num_boost_round=200)
print("LightGBM training complete.")

# 13) Predict & evaluate
y_pred_prob = bst.predict(X_test_proc)
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test_enc, y_pred)
prec = precision_score(y_test_enc, y_pred)
rec  = recall_score(y_test_enc, y_pred)
f1   = f1_score(y_test_enc, y_pred)
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print("\nClassification report:")
cr_text = classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4)
print(cr_text)

# Save classification report to file
with open(os.path.join(ARTIFACT_DIR, 'classification_report.txt'), 'w') as fh:
    fh.write(cr_text)
print("Saved classification report to artifacts.")

# Confusion matrix plot and save (after predictions)
cm = confusion_matrix(y_test_enc, y_pred)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight', dpi=150)
plt.close()
print("Saved confusion matrix to", cm_path)

# ------------------ Distribution plotting (bar + pie) ------------------
def plot_attack_distribution_from_series(y_series, out_dir, fname='attack_distribution.png', top_n=10):
    """
    Creates a side-by-side Bar Chart + Pie Chart showing attack category distribution.
    y_series : Pandas Series or list of labels (strings)
    out_dir  : directory where the image will be saved
    fname    : image name
    top_n    : how many categories to show in bar chart (others grouped as 'Other')
    """
    # Ensure Series type
    if not isinstance(y_series, pd.Series):
        y_series = pd.Series(y_series)

    # Count frequencies
    vc = y_series.value_counts()
    total = vc.sum()

    # Bar chart grouping
    if top_n is not None and top_n < len(vc):
        top = vc.iloc[:top_n]
        other = vc.iloc[top_n:].sum()
        vc_bar = pd.concat([top, pd.Series({'Other': other})])
    else:
        vc_bar = vc

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    sns.set_style("whitegrid")

    # ========== BAR CHART ==========
    ax = axes[0]
    bars = sns.barplot(x=vc_bar.values, y=vc_bar.index, ax=ax, palette="Blues_d")
    ax.set_xlabel('Count')
    ax.set_ylabel('Attack Type')
    ax.set_title('Attack Category Distribution (Training)')

    # Add text labels on bars
    for p, v in zip(bars.patches, vc_bar.values):
        w = p.get_width()
        ax.text(w + max(vc_bar.values) * 0.01, p.get_y() + p.get_height() / 2,
                f"{int(v):,}", va='center', fontsize=9)

    # ========== PIE CHART ==========
    ax2 = axes[1]
    labels = vc.index.tolist()
    sizes = vc.values
    explode = [0.03 if (s / total) < 0.03 else 0 for s in sizes]  # slightly highlight small slices

    ax2.pie(
        sizes,
        labels=labels,
        autopct=lambda pct_val: f"{pct_val:.1f}%" if pct_val >= 1 else '',
        startangle=140,
        pctdistance=0.8,
        labeldistance=1.05,
        explode=explode
    )

    ax2.axis('equal')
    ax2.set_title('Attack Category Percentage')

    # Save image
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print("Saved attack distribution figure to", out_path)
    return out_path

# Call distribution plot (use original multi-class label for distribution)
plot_attack_distribution_from_series(df_train[label_col].astype(str), ARTIFACT_DIR, fname='attack_distribution.png', top_n=10)

# 14) Save artifacts
joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, 'preprocessor_sparse.joblib'))
joblib.dump(le, os.path.join(ARTIFACT_DIR, 'label_encoder.joblib'))
bst.save_model(os.path.join(ARTIFACT_DIR, 'lgb_binary_model.txt'))
print("Artifacts saved to", ARTIFACT_DIR)

# 15) Sanity: warn if suspicious perfect score
if acc == 1.0:
    print("\nWARNING: Model achieved perfect accuracy (1.0). This is suspicious.")
    print("Check if any feature still leaks label or if train/test overlap exists.")
else:
    print("\nTraining finished. If you want improved recall/precision tune threshold, ensembling, or feature engineering.")
