"""
前列腺癌 CSS 生存预测网络计算器
RSF 模型 + Streamlit
本地: streamlit run app.py
HF: 推送到 Spaces
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

st.set_page_config(
    page_title="Prostate Cancer CSS Predictor",
    page_icon="🎗️",
    layout="wide",
)

# ===========================================================================
# 加载模型
# ===========================================================================
@st.cache_resource
def load_model():
    with open("RSF.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_reference_data():
    """加载训练集患者数据用于 KM 参考曲线"""
    df = pd.read_csv("Prostate_Cancer_Splitted.csv")
    df['css_event'] = (df['CSS_state'] == 1).astype(int)
    return df

rsf = load_model()
df_all = load_reference_data()
train = df_all[df_all['court'] == 'train'].copy()

# ===========================================================================
# 特征编码函数
# ===========================================================================
FEATURE_ORDER = [
    'Age', 'Race_Black', 'Race_Other', 'Marital_Unmarried',
    'T_Stage_T2', 'T_Stage_T3', 'T_Stage_T4',
    'Gleason_Mid', 'Gleason_High', 'PSA_log',
    'Surgery_RP_bin', 'Radiation_bin', 'Chemotherapy_bin',
]

def encode_patient(age, t_stage, gleason, psa, surgery_rp, radiation, chemo, race, marital):
    """将用户输入编码为 13 维特征向量"""
    feat = np.zeros(13, dtype=np.float32)

    feat[0] = age  # Age
    feat[1] = 1.0 if race == 'Black' else 0.0   # Race_Black
    feat[2] = 1.0 if race == 'Other' else 0.0   # Race_Other
    feat[3] = 1.0 if marital == 'Unmarried' else 0.0  # Marital_Unmarried

    t_map = {'T1': (0,0,0), 'T2': (0,1,2), 'T3': (3,4,5), 'T4': (6,7,8)}  # dummy
    if t_stage == 'T2':  feat[4] = 1.0
    elif t_stage == 'T3': feat[5] = 1.0
    elif t_stage == 'T4': feat[6] = 1.0

    if gleason == 'Grade Mid':      feat[7] = 1.0
    elif gleason == 'Grade High':   feat[8] = 1.0

    feat[9] = np.log(float(psa) + 1)  # PSA_log
    feat[10] = 1.0 if surgery_rp == 'Yes' else 0.0
    feat[11] = 1.0 if radiation == 'Yes' else 0.0
    feat[12] = 1.0 if chemo == 'Yes' else 0.0

    return feat.reshape(1, -1)

# ===========================================================================
# 训练集参考 KM 曲线
# ===========================================================================
CUTOFF = 50.59  # 训练集风险分中位数

@st.cache_data
def compute_ref_curves():
    """计算训练集高低危参考 KM 曲线"""
    # 用训练集预测风险分
    train_feat = encode_batch(train)
    risk_train = rsf.predict(train_feat)
    train['risk'] = risk_train

    curves = {}
    for grp, label in [('Low Risk', lambda x: x < CUTOFF), ('High Risk', lambda x: x >= CUTOFF)]:
        sub = train[label(train['risk'])]
        kmf = KaplanMeierFitter()
        kmf.fit(sub['Survival_Months'], sub['css_event'], label=grp)
        curves[grp] = {
            'times': kmf.survival_function_.index.values,
            'surv': kmf.survival_function_.values.flatten(),
            'ci_low': kmf.confidence_interval_.iloc[:, 0].values,
            'ci_high': kmf.confidence_interval_.iloc[:, 1].values,
        }
    return curves

def encode_batch(df_sub):
    feat = np.column_stack([
        df_sub['Age'].values,
        (df_sub['Race'] == 'Black').astype(int).values,
        (df_sub['Race'] == 'Other').astype(int).values,
        (df_sub['Marital_Status'] == 'Unmarried').astype(int).values,
        (df_sub['T_Stage'] == 'T2').astype(int).values,
        (df_sub['T_Stage'] == 'T3').astype(int).values,
        (df_sub['T_Stage'] == 'T4').astype(int).values,
        (df_sub['Gleason_Group'] == 'Grade_Mid').astype(int).values,
        (df_sub['Gleason_Group'] == 'Grade_High').astype(int).values,
        np.log(df_sub['PSA'].values + 1),
        (df_sub['Surgery_RP'] == 'Yes').astype(int).values,
        (df_sub['Radiation'] == 'Yes').astype(int).values,
        (df_sub['Chemotherapy'] == 'Yes').astype(int).values,
    ]).astype(np.float32)
    return feat

ref_curves = compute_ref_curves()

# ===========================================================================
# UI 侧边栏 — 输入
# ===========================================================================
st.title("🎗 Prostate Cancer CSS Survival Predictor")
st.markdown("**Random Survival Forest (RSF) model — Cancer-Specific Survival**")
st.markdown("---")

with st.sidebar:
    st.header("Patient Characteristics")

    age = st.slider("Age", 35, 90, 65, 1)
    t_stage = st.selectbox("T Stage", ['T1', 'T2', 'T3', 'T4'], index=1)
    gleason = st.selectbox("Gleason Grade", ['Grade Low', 'Grade Mid', 'Grade High'], index=1)
    psa = st.number_input("PSA (ng/mL)", min_value=0.1, max_value=200.0, value=10.0, step=0.5)
    surgery_rp = st.radio("Surgery (RP)", ['No', 'Yes'], horizontal=True)
    radiation = st.radio("Radiation", ['No', 'Yes'], horizontal=True)
    chemo = st.radio("Chemotherapy", ['No', 'Yes'], horizontal=True)
    race = st.selectbox("Race", ['White', 'Black', 'Other'])
    marital = st.selectbox("Marital Status", ['Married', 'Unmarried'])

    st.markdown("---")
    st.caption("RSF trained on n=4,780 prostate cancer patients")
    st.caption("Endpoint: Cancer-specific survival (CSS)")

# ===========================================================================
# 预测
# ===========================================================================
X_input = encode_patient(age, t_stage, gleason, psa, surgery_rp, radiation, chemo, race, marital)

risk_score = float(rsf.predict(X_input)[0])
risk_group = "High Risk" if risk_score >= CUTOFF else "Low Risk"

surv_fn = rsf.predict_survival_function(X_input, return_array=False)[0]
ANCHORS = [36, 60, 84]
surv_probs = {t: surv_fn(t) for t in ANCHORS}

# ===========================================================================
# 主面板
# ===========================================================================
col1, col2, col3, col4 = st.columns(4)

risk_color = "#B2182B" if risk_group == "High Risk" else "#21908C"
with col1:
    st.metric("RSF Risk Score", f"{risk_score:.1f}")
with col2:
    st.markdown(f"**Risk Group**: :{'red' if risk_group == 'High Risk' else 'green'}[**{risk_group}**]")
with col3:
    st.metric("3-year CSS", f"{surv_probs[36]*100:.1f}%")
with col4:
    st.metric(f"Cutoff = {CUTOFF}", f"Risk: {1-surv_probs[36]:.1%}")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("CSS Survival Probabilities")
    df_out = pd.DataFrame({
        "Time": ["3 years", "5 years", "7 years"],
        "CSS Probability": [f"{surv_probs[36]*100:.1f}%", f"{surv_probs[60]*100:.1f}%", f"{surv_probs[84]*100:.1f}%"],
        "Risk": [f"{1-surv_probs[t]:.1%}" for t in ANCHORS],
    })
    st.dataframe(df_out, hide_index=True, use_container_width=True)

    st.markdown(f"""
    | Variable | Value |
    |----------|-------|
    | Age | {age} |
    | T Stage | {t_stage} |
    | Gleason | {gleason.split()[-1]} |
    | PSA | {psa} ng/mL |
    | Surgery (RP) | {surgery_rp} |
    | Radiation | {radiation} |
    | Chemotherapy | {chemo} |
    | Race | {race} |
    | Marital | {marital} |
    """)

with col_b:
    st.subheader("Predicted CSS Survival Curve")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 患者预测曲线
    t_range = np.linspace(0, 180, 200)
    surv_curve = np.array([surv_fn(t) for t in t_range])
    ax.plot(t_range, surv_curve, color=risk_color, linewidth=2.5, label=f'Patient ({risk_group})')

    # 参考曲线
    for grp, color, lw in [('Low Risk', '#21908C', 1.2), ('High Risk', '#FDE725', 1.2)]:
        rc = ref_curves[grp]
        ax.plot(rc['times'], rc['surv'], color=color, linewidth=lw, linestyle='--', alpha=0.7, label=grp)

    # 锚点标注
    for t in ANCHORS:
        s = surv_fn(t)
        ax.scatter(t, s, color=risk_color, s=40, zorder=5, edgecolors='white', linewidth=0.5)
        ax.annotate(f'{s*100:.1f}%', (t+2, s+0.02), fontsize=8, fontweight='bold', color=risk_color)

    ax.set_xlim(0, 180); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Cancer-Specific Survival')
    ax.legend(fontsize=8, loc='lower left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

st.markdown("---")
st.caption("This tool is for research purposes only. Not for clinical use.")
