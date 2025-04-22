# hu_threshold_explorer.py
import numpy as np, pandas as pd, streamlit as st
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

# ---------- locate data file relative to this script ----------------
ROOT_DIR  = Path(__file__).parent           # repo root when deployed
DATA_PATH = ROOT_DIR / "refDEXA_and_HU.xlsx"

# ---------- constants -----------------------------------------------
COL_DXA_TERN   = "diagnosisdexa"
COL_DXA_OPBIN  = "wholedxa_op_vs_no_op"
COL_DXA_ABNBIN = "wholedxa_abnormal_vs_normal"
HU_COLS = [f"hu_{x}" for x in
           ("t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3", "l4")]

# ---------- cached loader -------------------------------------------
@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_"))
    if "median_hu" not in df.columns:
        df["median_hu"] = df[HU_COLS].median(axis=1, skipna=True)
    return df.dropna(subset=["median_hu", COL_DXA_TERN,
                             COL_DXA_OPBIN, COL_DXA_ABNBIN])

df = load_df(DATA_PATH)
st.sidebar.success(f"Loaded {len(df):,} rows")

# ---------- threshold controls --------------------------------------
st.sidebar.header("HU thresholds")
op_T  = st.sidebar.slider("Osteoporosis HU", 55, 120, 110)
pen_T = st.sidebar.slider("Osteopenia HU",  100, 170, 160)
if op_T >= pen_T:
    st.error("Osteoporosis HU must be lower than osteopenia HU"); st.stop()

# ---------- classify -------------------------------------------------
ct_class = np.select(
    [df["median_hu"] <= op_T, df["median_hu"] <= pen_T],
    [2, 1], default=0
)
df["ct_bin_op"]  = (ct_class == 2).astype(int)
df["ct_bin_abn"] = (ct_class >= 1).astype(int)

def metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else np.nan
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    ppv  = tp/(tp+fp) if (tp+fp) else np.nan
    npv  = tn/(tn+fn) if (tn+fn) else np.nan
    acc  = (tp+tn)/(tp+tn+fp+fn)
    auc  = roc_auc_score(y_true, score)
    return dict(Sensitivity=sens, Specificity=spec,
                PPV=ppv, NPV=npv, Accuracy=acc, AUC=auc)

m_op  = metrics(df[COL_DXA_OPBIN],  df["ct_bin_op"],  ct_class)
m_abn = metrics(df[COL_DXA_ABNBIN], df["ct_bin_abn"], ct_class)

# ---------- display --------------------------------------------------
st.title("HU Threshold Explorer (Whole‑DXA reference)")
st.write(f"Osteoporosis < **{op_T} HU**   Osteopenia < **{pen_T} HU**")

c1, c2 = st.columns(2)
with c1: st.subheader("Osteoporosis vs No‑OP"); st.table(m_op)
with c2: st.subheader("Abnormal vs Normal");    st.table(m_abn)

label_map = {2:"OSTEOPOROSIS",1:"OSTEOPENIA",0:"NORMAL"}
cont = pd.crosstab(
    df[COL_DXA_TERN].map(label_map),
    ct_class.map(lambda x: "CT_"+label_map[x]),
    rownames=["Whole DXA"], colnames=["CT"], dropna=False
).reindex(index=[label_map[i] for i in (2,1,0)],
          columns=[f"CT_{label_map[i]}" for i in (2,1,0)])
st.subheader("3 × 3 Contingency Table")
st.table(cont)
