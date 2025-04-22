# hu_threshold_explorer.py  (Google‑Drive version)
import io, requests, gdown, numpy as np, pandas as pd, streamlit as st
from sklearn.metrics import roc_auc_score, confusion_matrix

# ------------ column names ---------------------------------------------------
COL_DXA_TERN   = "diagnosisdexa"
COL_DXA_OPBIN  = "wholedxa_op_vs_no_op"
COL_DXA_ABNBIN = "wholedxa_abnormal_vs_normal"
HU_COLS = [f"hu_{x}" for x in
           ("t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3", "l4")]

# ------------ cached loader --------------------------------------------------
@st.cache_data
def load_df() -> pd.DataFrame:
    # → secrets must contain [data] gdrive_file_id, file_ext
    sec = st.secrets["data"]
    file_id  = sec["gdrive_file_id"]
    ext      = sec.get("file_ext", "parquet")

    if ext == "parquet":
        tmp = gdown.download(id=file_id, quiet=True, fuzzy=True)
        df = pd.read_parquet(tmp)
    else:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        buf = io.BytesIO(requests.get(url, timeout=60).content)
        df  = pd.read_excel(buf)                 # needs openpyxl
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_"))
    if "median_hu" not in df.columns:
        df["median_hu"] = df[HU_COLS].median(axis=1, skipna=True)
    return df.dropna(subset=["median_hu",
                             COL_DXA_TERN, COL_DXA_OPBIN, COL_DXA_ABNBIN])

df = load_df()
st.sidebar.success(f"Dataset loaded: {len(df):,} rows")

# ------------ HU threshold controls -----------------------------------------
st.sidebar.header("HU thresholds")
op_T  = st.sidebar.slider("Osteoporosis HU (≤)", 55, 120, 110)
pen_T = st.sidebar.slider("Osteopenia HU (≤)",   100, 170, 160)
if op_T >= pen_T:
    st.sidebar.error("Osteoporosis HU must be lower than osteopenia HU")
    st.stop()

# ------------ classify CT ----------------------------------------------------
ct_diag = np.select(
    [df["median_hu"] <= op_T,
     df["median_hu"] <= pen_T],
    [2, 1], default=0
)
df["ct_diag"]    = ct_diag
df["ct_bin_op"]  = (ct_diag == 2).astype(int)
df["ct_bin_abn"] = (ct_diag >= 1).astype(int)

# ------------ metric helper --------------------------------------------------
def binary_metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens, spec = tp/(tp+fn), tn/(tn+fp)
    ppv, npv   = tp/(tp+fp), tn/(tn+fn)
    acc        = (tp+tn)/(tp+tn+fp+fn)
    auc        = roc_auc_score(y_true, score)
    return dict(Sensitivity=sens, Specificity=spec,
                PPV=ppv, NPV=npv, Accuracy=acc, AUC=auc)

metrics_op  = binary_metrics(df[COL_DXA_OPBIN],  df["ct_bin_op"],  ct_diag)
metrics_abn = binary_metrics(df[COL_DXA_ABNBIN], df["ct_bin_abn"], ct_diag)

# ------------ contingency table ---------------------------------------------
label = {2:"OSTEOPOROSIS",1:"OSTEOPENIA",0:"NORMAL"}
cont = pd.crosstab(
    df[COL_DXA_TERN].map(label),
    df["ct_diag"].map(lambda x: "CT_"+label[x]),
    rownames=["Whole DXA"], colnames=["CT"], dropna=False
).reindex(index=[label[i] for i in (2,1,0)],
          columns=[f"CT_{label[i]}" for i in (2,1,0)])

# ------------ display --------------------------------------------------------
st.title("HU Threshold Explorer – Whole‑body DXA")
st.markdown(
    f"Osteoporosis ≤ **{op_T} HU**   ·   "
    f"Osteopenia ≤ **{pen_T} HU**  (and > {op_T})"
)

c1, c2 = st.columns(2)
with c1: st.subheader("Osteoporosis vs No‑OP"); st.table({k:f"{v:.3f}" for k,v in metrics_op.items()})
with c2: st.subheader("Abnormal vs Normal");    st.table({k:f"{v:.3f}" for k,v in metrics_abn.items()})

st.subheader("3 × 3 Contingency Table")
st.table(cont)
