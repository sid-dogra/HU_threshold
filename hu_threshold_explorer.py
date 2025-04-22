# hu_threshold_explorer.py
# --------------------------------------------------------------
# Streamlit app to explore opportunistic CT HU thresholds
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

# -------- CONFIG ------------------------------------------------
DATA_PATH = Path(
    "/Users/siddhantdogra/Documents/Research/Active_Research/"
    "Opportunistic Screening/Spine DXA vs CT/refDEXA_and_HU.xlsx"
)

# DXA reference columns (whole‑body only)
COL_DXA_TERN   = "diagnosisdexa"                 # 0/1/2
COL_DXA_OPBIN  = "wholedxa_op_vs_no_op"          # 1 = osteoporosis
COL_DXA_ABNBIN = "wholedxa_abnormal_vs_normal"   # 1 = osteopenia + osteoporosis

# HU columns present in the sheet (adjust if needed)
HU_COLS = ["hu_t8", "hu_t9", "hu_t10", "hu_t11", "hu_t12",
           "hu_l1", "hu_l2", "hu_l3", "hu_l4"]

# -------- LOAD & PRE‑PROCESS -----------------------------------
@st.cache_data
def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_"))
    if "median_hu" not in df.columns:
        df["median_hu"] = df[HU_COLS].median(axis=1, skipna=True)
    # drop any row lacking HU or DXA refs
    df = df.dropna(subset=["median_hu",
                           COL_DXA_TERN, COL_DXA_OPBIN, COL_DXA_ABNBIN])
    df["median_hu"] = df["median_hu"].astype(float)
    return df

df = load_dataframe(DATA_PATH)
st.sidebar.success(f"Dataset loaded: {len(df):,} subjects")

# -------- THRESHOLD INPUTS -------------------------------------
st.sidebar.header("Set HU thresholds")

op_T  = st.sidebar.slider(
    "Osteoporosis cut‑off (HU)",
    min_value=55, max_value=120, value=110, step=1
)
pen_T = st.sidebar.slider(
    "Osteopenia upper bound (HU)",
    min_value=100, max_value=170, value=160, step=1
)

if op_T >= pen_T:
    st.sidebar.error("Osteoporosis cut‑off must be lower than the osteopenia bound.")
    st.stop()

# -------- CLASSIFY CT ------------------------------------------
ct_class = np.select(
    [df["median_hu"] <= op_T,
     df["median_hu"] <= pen_T],
    [2, 1], default=0
)  # 2 = osteoporosis, 1 = osteopenia, 0 = normal

df["ct_diag"]                = ct_class
df["ct_op_vs_no_op"]         = (ct_class == 2).astype(int)
df["ct_abnormal_vs_normal"]  = (ct_class >= 1).astype(int)

# -------- METRIC HELPER ----------------------------------------
def binary_metrics(y_true: pd.Series, y_pred: pd.Series, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn)
    auc  = roc_auc_score(y_true, score)
    return {
        "Sensitivity": sens, "Specificity": spec,
        "PPV": ppv, "NPV": npv,
        "Accuracy": acc, "AUC": auc
    }

# -------- COMPUTE METRICS --------------------------------------
metrics_op  = binary_metrics(
    df[COL_DXA_OPBIN].astype(int),
    df["ct_op_vs_no_op"],
    ct_class
)
metrics_abn = binary_metrics(
    df[COL_DXA_ABNBIN].astype(int),
    df["ct_abnormal_vs_normal"],
    ct_class
)

# -------- CONTINGENCY TABLE ------------------------------------
label_map = {2: "OSTEOPOROSIS", 1: "OSTEOPENIA", 0: "NORMAL"}
rows = [label_map[i] for i in (2, 1, 0)]
cols = [f"CT_{label_map[i]}" for i in (2, 1, 0)]

cont_table = (
    pd.crosstab(
        df[COL_DXA_TERN].map(label_map),
        ct_class.map(lambda x: f"CT_{label_map[x]}"),
        rownames=["Whole DXA"], colnames=["CT"], dropna=False
    )
    .reindex(index=rows, columns=cols)
)

# -------- DISPLAY ----------------------------------------------
st.title("HU Threshold Explorer — Whole‑Body DXA Reference")

st.markdown(f"""
**Current thresholds**

* Osteoporosis: HU < **{op_T}**
* Osteopenia:  HU < **{pen_T}** (and ≥ {op_T})
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Osteoporosis vs Non‑osteoporosis")
    st.table({k: f"{v:.3f}" for k, v in metrics_op.items()})

with col2:
    st.subheader("Abnormal vs Normal")
    st.table({k: f"{v:.3f}" for k, v in metrics_abn.items()})

st.subheader("3 × 3 Contingency Table (Whole‑DXA vs CT)")
st.table(cont_table)
