import streamlit as st
import joblib   ##load model files
import os ##make file paths
import numpy as np ##numeric calculations
import pandas as pd ##Dtaframe , tables
import scipy.sparse as sp   ##handle sparse metrics
import lightgbm as lgb  ##predict machine learning model
import matplotlib.pyplot as plt  ##for basic charts
import seaborn as sns   ##for beautiful and stylish charts



#----------page title--------#

st.set_page_config(page_title="Intrusion Detection System Predictor", layout="wide")
st.title("🔐 Intrusion Detection System Predictor")


##---------Config and artifact paths--------#
ARTIFACT_DIR = st.sidebar.text_input("Artifacts directory", value="artifacts")
PREPROC_NAME = "preprocessor_sparse.joblib"
LABELENC_NAME = "label_encoder.joblib"
LGB_MODEL_NAME = "lgb_binary_model.txt"
MEDIANS_NAME = "train_medians.joblib"


preproc_path = os.path.join(ARTIFACT_DIR, PREPROC_NAME)
labelenc_path = os.path.join(ARTIFACT_DIR, LABELENC_NAME)
lgb_model_path = os.path.join(ARTIFACT_DIR, LGB_MODEL_NAME)
medians_path = os.path.join(ARTIFACT_DIR, MEDIANS_NAME)


st.sidebar.markdown("### Model artifacts")
st.sidebar.write(f"- Preprocessor: `{PREPROC_NAME}`")
st.sidebar.write(f"- Label encoder: `{LABELENC_NAME}`")
st.sidebar.write(f"- LightGBM model: `{LGB_MODEL_NAME}`")
st.sidebar.write("Change artifact directory above if needed.")


##---------Helper: load artifacts----------#
@st.cache_resource
def load_artifacts(preproc_p, labelenc_p, model_p, medians_p=None):
    if not os.path.exists(preproc_p) or not os.path.exists(labelenc_p) or not os.path.exists(model_p):
        raise FileNotFoundError("Required artifact not found.Make sure preprocessor, label enocder and model exist.")
    preprocessor = joblib.load(preproc_p)
    label_enc = joblib.load(labelenc_p)
    ##LightGBM: try load as Booster
    try:
        lgb_booster = lgb.Booster(model_file = model_p)
    except Exception:
        ##may be saved with joblib
        m = joblib.load(model_p)
        if isinstance(m, lgb.basic.Booster):
            lgb_booster = m
        else:
            raise
    train_medians = None
    if medians_p and os.path.exists(medians_p):
        try:
           train_medians = joblib.load(medians_p) 
        except Exception:
            train_medians = None
    return preprocessor, label_enc, lgb_booster, train_medians

## try to load; show helpful message if missing
try:
    preprocessor, label_enc, lgb_model, train_medians = load_artifacts(preproc_path, labelenc_path, lgb_model_path, medians_path)
    st.sidebar.success("Artifacts loaded.")
except Exception as e:
    st.error(f"Artifacts load error: {e}")
    st.stop()


##Determine expected input feature columns (preprocessor input)
if hasattr(preprocessor, "feature_names_in_"):
    feature_names = list(preprocessor.feature_names_in_)
else:
    ##falllback: trry to collect raw columns from transformers
    try:
        feature_names = []
        for t in preprocessor.transformers_:
            if isinstance(t, tuple) and len(t) >= 3:
                cols = t[2]
                if isinstance(cols, (list, tuple)):
                    feature_names.extend(list(cols))
        feature_names = list(dict.fromkeys(feature_names))
    except Exception:
        st.error("Cannot determine preprocessor input feature names.Ensure preprocessor exposes feature_names_in_.")
        st.stop()


st.sidebar.write(f"Detected {len(feature_names)} input features.")


##-----Alias map and parser------------##

ALIAS_MAP = {
    "vdur": "dur",
    "duration": "dur",
    "src_bytes": "sbytes",
    "dst_bytes": "dbytes",
    "src_pkts": "spkts",
    "dst_pkts": "dpkts",
    # add more aliases if you like
}


def parse_input_block(text: str) -> dict:
    values = {}
    if not text or not text.strip():
        return values
    parts = [p.strip() for p in text.replace("\n", " ").split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v =p.split("=", 1)
            values[k.strip()] = v.strip()
    ## apply aliases
    out = {}
    for k, v in values.items():
        k_l = k.lower()
        if k_l in ALIAS_MAP:
            out_key = ALIAS_MAP[k_l]
        else:
            out_key = k.strip()
        out[out_key] = v
    return out

def coerce_value(v):
    try:
        if isinstance(v, str) and v.strip() == "":
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return v
        vs = v.replace(",", "")
        return float(vs) if '.' in vs or vs.isdigit() else v
    except Exception :
        return v
    

#------------------Utility:  convert output to dense(for display)------------------#
def dense_from_proc(X_proc):
    try:
        return X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)
    except Exception:
        return None
    

##-------------Main UI------------#
mode = st.radio("Mode", ["Single prediction", "Batch CSV prediction"], horizontal=True)


## proability threshold slider

threshold = st.sidebar.slider("Probability threshold for class 'Attack'", 0.0, 1.0, 0.5, 0.01)

##------------Single prediction------------------#
if mode.startswith("Single"):
    st.header("Single prediction")
    st.markdown("Paste key=value pairs separated by commas. Example:")
    st.code("dur=0.35, proto=tcp, sbytes=350, dbytes=1200, spkts=5, dpkts=5")
    txt = st.text_area("Input (key=value, comma separated):", height=140)

    if st.button("Parse & Predict"):
        parsed = parse_input_block(txt)
        if not parsed:
            st.error("No parsed input. Please paste key=value pairs.")
        else:
            # build full row
            row = {c: np.nan for c in feature_names}
            for k, v in parsed.items():
                if k in row:
                    row[k] = coerce_value(v)
            df_row = pd.DataFrame([row], columns=feature_names)

            # fill numeric missing with train medians if available
            if train_medians is not None:
                for c in df_row.columns:
                    if pd.isna(df_row.loc[0, c]) and c in train_medians.index:
                        df_row.loc[0, c] = train_medians[c]
                st.info("Missing numeric features filled from training medians (where available).")

            # --- FIRST: show final input row (what will be fed to preprocessor) ---
            st.subheader("Final input row (non-empty cols)")
            non_empty = df_row.loc[:, df_row.notna().any()]
            if non_empty.shape[1] == 0:
                st.write("All columns empty — preprocessor will impute defaults.")
            else:
                st.dataframe(non_empty.T)

            # --- NEXT: preprocess and show processed vector ---
            try:
                X_proc = preprocessor.transform(df_row)
                dense = dense_from_proc(X_proc)
                # show processed features (if possible)
                if dense is not None:
                    try:
                        cols_out = preprocessor.get_feature_names_out()
                    except Exception:
                        cols_out = [f"f{i}" for i in range(dense.shape[1])]
                    proc_df = pd.DataFrame(dense[0].reshape(1, -1), columns=cols_out)
                    st.subheader("Processed feature vector (first row)")
                    st.dataframe(proc_df.T)
                else:
                    st.write("Processed shape:", getattr(X_proc, "shape", None))

                # --- NOW show parsed alias mapping (moved after the processed view) ---
                st.subheader("Parsed (after alias mapping)")
                st.json(parsed)

                # predict using LightGBM booster
                # LightGBM expects 2D array (dense) or scipy CSR; use dense for single row
                X_for_pred = dense if dense is not None else X_proc
                probs = lgb_model.predict(X_for_pred)
                # probs is probability of positive class (depending on encoding)
                # Determine label mapping from label_enc
                classes = list(label_enc.classes_)
                # label_enc likely encodes classes lexicographically. We want probability of 'Attack'
                # If label_enc.classes_ == ['Attack','Normal'] or vice versa, find index of 'Attack'
                if 'Attack' in classes:
                    attack_idx = classes.index('Attack')
                    # If model returns prob of positive class only (binary), it returns prob for class 1 (positive)
                    # LightGBM.booster.predict returns prob of positive unless configured differently.
                    # We'll assume probs is prob of positive class (class index 1), but safer: check length.
                    if isinstance(probs, np.ndarray) and probs.ndim == 1:
                        prob_attack = float(probs[0])
                    else:
                        # fallback: if 2-D (n_samples x n_classes)
                        prob_attack = float(probs[0][attack_idx])
                else:
                    # if no 'Attack' label (unlikely), pick index 1 as positive
                    if isinstance(probs, np.ndarray) and probs.ndim == 1:
                        prob_attack = float(probs[0])
                    else:
                        prob_attack = float(probs[0][1] if probs.shape[1] > 1 else probs[0][0])

                pred_label = "Attack" if prob_attack >= threshold else "Normal"

                st.success(f"Prediction: **{pred_label}**  (prob Attack = {prob_attack:.4f})")

                # small bar chart
                fig, ax = plt.subplots(figsize=(6,2))
                sns.barplot(x=[prob_attack, 1-prob_attack], y=['Attack', 'Normal'], ax=ax)
                ax.set_xlim(0,1)
                st.pyplot(fig)

                # show raw probability array if multi-d
                if isinstance(probs, np.ndarray) and probs.ndim == 2:
                    prob_df = pd.DataFrame(probs[0], index=classes, columns=['prob']).sort_values('prob', ascending=False)
                    st.subheader("Full class probabilities")
                    st.dataframe(prob_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------
# BATCH CSV PREDICTION
# -----------------------
else:
    st.header("Batch CSV prediction")
    st.markdown("Upload CSV file. Columns should match preprocessor input names (or missing columns will be added as NaN).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            st.subheader("Preview (first rows)")
            st.dataframe(df.head())

            if st.button("Run batch prediction"):
                # add missing columns
                for c in feature_names:
                    if c not in df.columns:
                        df[c] = np.nan

                df_in = df[feature_names].copy()

                # fill numeric missing with medians if available
                if train_medians is not None:
                    for c in df_in.columns:
                        if c in train_medians.index:
                            df_in[c] = df_in[c].fillna(train_medians[c])

                # preprocess
                try:
                    X_proc = preprocessor.transform(df_in)
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    X_proc = None

                if X_proc is not None:
                    # LightGBM predict: accept sparse/dense
                    # if X_proc is sparse, convert to array for predict (light on memory for batch)
                    try:
                        X_for_pred = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)
                    except Exception:
                        X_for_pred = X_proc

                    probs = lgb_model.predict(X_for_pred)
                    # probs: if shape (n_samples,), it's prob positive; if (n_samples, n_classes) then get column for 'Attack' index
                    classes = list(label_enc.classes_)
                    if isinstance(probs, np.ndarray) and probs.ndim == 1:
                        prob_attack_arr = probs
                    else:
                        if 'Attack' in classes:
                            attack_idx = classes.index('Attack')
                            prob_attack_arr = np.array([p[attack_idx] for p in probs])
                        else:
                            # fallback: take column 1 if exists
                            prob_attack_arr = np.array([p[1] if len(p)>1 else p[0] for p in probs])

                    pred_arr = np.where(prob_attack_arr >= threshold, "Attack", "Normal")
                    out_df = df.copy()
                    out_df["prob_attack"] = prob_attack_arr
                    out_df["predicted"] = pred_arr

                    st.success("Batch prediction finished.")
                    st.dataframe(out_df.head(50))

                    # download
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

                    # distribution plot
                    st.subheader("Prediction distribution")
                    fig2, ax2 = plt.subplots(figsize=(8,3))
                    sns.countplot(x=out_df["predicted"], order=out_df["predicted"].value_counts().index, ax=ax2)
                    st.pyplot(fig2)

# -----------------------
# Footer: helpful info
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Notes")
st.sidebar.markdown("- The app expects artifacts saved by training script in the `artifacts` directory by default.")
st.sidebar.markdown("- Single-row mode fills missing numeric values from training medians if available; otherwise preprocessor imputation used.")
st.sidebar.markdown("- For SHAP explainability, install `shap` and we can extend the UI to show per-sample attribution.")
