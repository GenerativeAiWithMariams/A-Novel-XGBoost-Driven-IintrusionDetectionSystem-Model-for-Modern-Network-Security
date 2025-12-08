# train_multiclass_improved.py
# Usage examples:
#   python train_multiclass_improved.py
#   python train_multiclass_improved.py --use_gbac
#   python train_multiclass_improved.py --merge_rare
#
# Requirements:
#   pip install numpy pandas scipy scikit-learn xgboost lightgbm joblib matplotlib seaborn imbalanced-learn

import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ------------ Arguments ------------
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='UNSW_NB15_training-set.csv')
parser.add_argument('--test', type=str, default='UNSW_NB15_testing-set.csv')
parser.add_argument('--artifacts', type=str, default='artifacts_improved')
parser.add_argument('--use_gbac', action='store_true', help='Use GBAC iterative weighting (extra time)')
parser.add_argument('--merge_rare', action='store_true', help='Merge extremely rare classes into "Other"')
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

TRAIN_CSV = args.train
TEST_CSV = args.test
ARTIFACT_DIR = args.artifacts
os.makedirs(ARTIFACT_DIR, exist_ok=True)
RANDOM_STATE = args.random_state

# ------------ Load data ------------
print("Loading CSVs...")
df_train = pd.read_csv(TRAIN_CSV, low_memory=True)
df_test  = pd.read_csv(TEST_CSV,  low_memory=True)

# detect label
if 'attack_cat' in df_train.columns:
    label_col = 'attack_cat'
elif 'label' in df_train.columns:
    label_col = 'label'
else:
    raise RuntimeError("Label column not found in CSVs.")

print("Label column:", label_col)

# drop identifiers
drop_cols = [c for c in ['id','ID','flow_id','timestamp','start_time','start','src_ip','dst_ip','sport','dport'] if c in df_train.columns]
df_train = df_train.drop(columns=drop_cols, errors='ignore')
df_test  = df_test.drop(columns=drop_cols, errors='ignore')

# Optionally merge very rare classes into "Other"
if args.merge_rare:
    # threshold absolute count in train; classes below this will be merged
    MERGE_COUNT = 200  # you can change
    vc = df_train[label_col].value_counts()
    rare = vc[vc < MERGE_COUNT].index.tolist()
    if len(rare) > 0:
        print("Merging rare classes into 'Other':", rare)
        df_train[label_col] = df_train[label_col].astype(str).apply(lambda x: 'Other' if x in rare else x)
        df_test[label_col]  = df_test[label_col].astype(str).apply(lambda x: 'Other' if x in rare else x)

# align features
feature_cols = [c for c in df_train.columns if c in df_test.columns and c != label_col]
X_train = df_train[feature_cols].copy(); y_train = df_train[label_col].copy()
X_test  = df_test[feature_cols].copy(); y_test  = df_test[label_col].copy()

# numeric / categorical split
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Feature counts — numeric:", len(num_cols), "categorical:", len(cat_cols))

# Preprocessor (sparse OHE for categories)
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)], remainder='drop', sparse_threshold=0.0)
print("Fitting preprocessor...")
preprocessor.fit(X_train)

# helper to build sparse CSR matrix
def transform_to_csr(preproc, X, num_cols, cat_cols):
    X_num = preproc.named_transformers_['num'].transform(X[num_cols]) if num_cols else None
    cat_imputer = preproc.named_transformers_['cat'].named_steps['imputer'] if cat_cols else None
    cat_ohe = preproc.named_transformers_['cat'].named_steps['ohe'] if cat_cols else None
    X_cat = None
    if cat_cols:
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

# Label encode
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.astype(str))
y_test_enc  = le.transform(y_test.astype(str))
num_class = len(le.classes_)
print("Detected classes:", num_class, le.classes_)

# Compute initial inverse-frequency sample weights
class_counts = Counter(y_train_enc)
n = len(y_train_enc)
class_weight = {cls: n/(len(class_counts)*count) for cls, count in class_counts.items()}
sample_weights = np.array([class_weight[int(lbl)] for lbl in y_train_enc], dtype=float)
print("Sample weights stats:", sample_weights.min(), sample_weights.max())

# Optional GBAC iterative adjustment
if args.use_gbac:
    print("Running GBAC iterative weighting (2 iterations)...")
    for it in range(2):   #3loop run 2times
        dtmp = xgb.DMatrix(X_train_proc, label=y_train_enc, weight=sample_weights)
        params_tmp = {'objective':'multi:softmax', 'num_class':num_class, 'eta':0.3, 'max_depth':3, 'seed':RANDOM_STATE}
        bst_tmp = xgb.train(params_tmp, dtmp, num_boost_round=30, verbose_eval=False)
        preds_tmp = bst_tmp.predict(dtmp).astype(int)
        perr = {}
        for cls in np.unique(y_train_enc):
            idx = np.where(y_train_enc == cls)[0]
            if len(idx)==0: continue
            perr[cls] = 1.0 - (preds_tmp[idx] == y_train_enc[idx]).mean()
        for cls, err in perr.items():
            sample_weights[y_train_enc == cls] *= (1.0 + err)
        sample_weights = sample_weights * (len(sample_weights)/sample_weights.sum())
    print("GBAC done. Sample weights stats:", sample_weights.min(), sample_weights.max())

# Split train/val for early stopping
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train_proc, y_train_enc, sample_weights, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_enc)

dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
dtest = xgb.DMatrix(X_test_proc, label=y_test_enc)

# XGBoost params (gentler defaults)
params = {
    'objective': 'multi:softprob',
    'num_class': num_class,
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'seed': RANDOM_STATE
}

print("Training XGBoost with early stopping...")
bst = xgb.train(params, dtrain, num_boost_round=1000,
                evals=[(dtrain,'train'), (dval,'val')],
                early_stopping_rounds=50,
                verbose_eval=50)

# Predict & evaluate
y_prob = bst.predict(dtest)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test_enc, y_pred)
macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
print(f"Multiclass Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
print("Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4))

# Confusion matrix plot and save
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight', dpi=150)
plt.close()
print("Saved confusion matrix to", cm_path)
##Image distribution.png
# ------------------ Distribution plotting (bar + pie) ------------------

# Ensure matplotlib works in all environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# ------------------ CALL FUNCTION ------------------

# Safe default top_n instead of args.top_n_plot
plot_attack_distribution_from_series(
    y_train.astype(str),
    ARTIFACT_DIR,
    fname='attack_distribution.png',
    top_n=10   # Change this to any number you like
)

# Save artifacts
joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, 'preprocessor.joblib'))
joblib.dump(le, os.path.join(ARTIFACT_DIR, 'label_encodeR.joblib'))
bst.save_model(os.path.join(ARTIFACT_DIR, 'xgb_multiclass.model'))
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(os.path.join(ARTIFACT_DIR, 'confusion_matrix.csv'))
print("Saved artifacts to", ARTIFACT_DIR)

print("Done.")
