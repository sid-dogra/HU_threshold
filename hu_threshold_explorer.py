# hu_threshold_explorer.py
# --------------------------------------------------------------
# Streamlit app: explore HU thresholds vs whole‑body DXA
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

# ---------- locate dataset (relative path) --------------------
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "refDEXA_and_HU.xlsx"           # keep in repo root

# ---------- DXA columns ---------------------------------------
COL_DXA_TERN   = "diagnosisdexa"                 # 0 / 1 / 2
COL_DXA_OPBIN  = "wholedxa_op_vs_no_op"          # 1 = osteoporosis
COL_DXA_ABNBIN = "wholedxa_abnormal_vs_normal"   # 1 = osteopenia + osteoporosis

HU_COLS = [f"hu_{x}" for x in
           ("t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3", "l4")]

# ---------- cached loader: fetch once, cache in RAM -----------------
import gdown, io, requests

@st.cache_data
def load_df() -> pd.DataFrame:
    file_id  = st.secrets["data"]["gdrive_file_id"]
    ext      = st.secrets["data"].get("file_ext", "parquet")

    if ext == "parquet":
        tmp_path = gdown.download(id=file_id, quiet=True, fuzzy=True)
        df = pd.read_parquet(tmp_path)
    else:  # assume .xlsx
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        content = requests.get(url, timeout=60).content
        df = pd.read_excel(io.BytesIO(content))   # needs openpyxl
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_"))
    if "median_hu" not in df.columns:
        hu_cols = [c for c in df.columns if c.startswith("hu_")]
        df["median_hu"] = df[hu_cols].median(axis=1, skipna=True)
    return df.dropna(subset=["median_hu",
                             COL_DXA_TERN, COL_DXA_OPBIN, COL_DXA_ABNBIN])

# ---------- classify CT ---------------------------------------
ct_diag = np.select(
    [df["median_hu"] <= op_T,
     df["median_hu"] <= pen_T],
    [2, 1], default=0
)
df["ct_diag"]               = ct_diag
df["ct_bin_op"]             = (ct_diag == 2).astype(int)
df["ct_bin_abn"]            = (ct_diag >= 1).astype(int)

# ---------- metric helper -------------------------------------
def binary_metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn)
    auc  = roc_auc_score(y_true, score)
    return dict(Sensitivity=sens, Specificity=spec,
                PPV=ppv, NPV=npv, Accuracy=acc, AUC=auc)

metrics_op  = binary_metrics(df[COL_DXA_OPBIN],  df["ct_bin_op"],  ct_diag)
metrics_abn = binary_metrics(df[COL_DXA_ABNBIN], df["ct_bin_abn"], ct_diag)

# ---------- contingency table ---------------------------------
label_map = {2: "OSTEOPOROSIS", 1: "OSTEOPENIA", 0: "NORMAL"}
cont_table = (
    pd.crosstab(
        df[COL_DXA_TERN].map(label_map),
        df["ct_diag"].map(lambda x: "CT_" + label_map[x]),
        rownames=["Whole DXA"], colnames=["CT"], dropna=False
    )
    .reindex(index=[label_map[i] for i in (2, 1, 0)],
             columns=[f"CT_{label_map[i]}" for i in (2, 1, 0)])
)

# ---------- display -------------------------------------------
st.title("HU Threshold Explorer  –  Whole‑body DXA Reference")
st.markdown(
    f"Osteoporosis ≤ **{op_T} HU**   ·   "
    f"Osteopenia ≤ **{pen_T} HU** (and > {op_T})"
)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Osteoporosis vs No‑OP")
    st.table({k: f"{v:.3f}" for k, v in metrics_op.items()})
with col2:
    st.subheader("Abnormal vs Normal")
    st.table({k: f"{v:.3f}" for k, v in metrics_abn.items()})

st.subheader("3 × 3 Contingency Table (Whole‑DXA vs CT)")
st.table(cont_table)
