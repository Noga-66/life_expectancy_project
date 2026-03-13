import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Life Expectancy Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS + Animations ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', 'Courier New', monospace !important; }

@keyframes fadeSlideDown { from{opacity:0;transform:translateY(-24px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeSlideUp   { from{opacity:0;transform:translateY(24px)}  to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn        { from{opacity:0} to{opacity:1} }
@keyframes scaleIn       { from{opacity:0;transform:scale(0.93)} to{opacity:1;transform:scale(1)} }
@keyframes shimmer       { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
@keyframes pulseGlow     { 0%,100%{box-shadow:0 0 0 0 rgba(200,169,110,0)} 50%{box-shadow:0 0 18px 4px rgba(200,169,110,0.18)} }
@keyframes borderPulse   { 0%,100%{border-color:rgba(200,169,110,0.15)} 50%{border-color:rgba(200,169,110,0.45)} }
@keyframes scanLine      { 0%{top:-4px} 100%{top:100%} }
@keyframes floatOrb      { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(30px,-20px) scale(1.05)} 66%{transform:translate(-20px,15px) scale(0.97)} }
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes slideInLeft   { from{opacity:0;transform:translateX(-32px)} to{opacity:1;transform:translateX(0)} }
@keyframes slideInRight  { from{opacity:0;transform:translateX(32px)}  to{opacity:1;transform:translateX(0)} }
@keyframes ripple        { 0%{transform:scale(0.95);box-shadow:0 0 0 0 rgba(200,169,110,0.25)} 70%{transform:scale(1);box-shadow:0 0 0 12px rgba(200,169,110,0)} 100%{transform:scale(0.95);box-shadow:0 0 0 0 rgba(200,169,110,0)} }
@keyframes countUp       { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

.stApp { background: #08080f; }
.stApp::before {
    content:''; position:fixed; top:-20%; left:-10%; width:55vw; height:55vw; border-radius:50%;
    background:radial-gradient(circle,rgba(200,169,110,0.055) 0%,transparent 65%);
    animation:floatOrb 14s ease-in-out infinite; pointer-events:none; z-index:0;
}
.stApp::after {
    content:''; position:fixed; bottom:-20%; right:-10%; width:50vw; height:50vw; border-radius:50%;
    background:radial-gradient(circle,rgba(78,205,196,0.04) 0%,transparent 65%);
    animation:floatOrb 18s ease-in-out infinite reverse; pointer-events:none; z-index:0;
}

#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem 4rem !important;max-width:1300px !important;position:relative;z-index:1;}

.hero-banner {
    background:linear-gradient(135deg,rgba(200,169,110,0.07) 0%,rgba(78,205,196,0.04) 100%);
    border:1px solid rgba(200,169,110,0.15); border-radius:4px; padding:36px 40px;
    margin-bottom:32px; position:relative; overflow:hidden;
    animation:fadeSlideDown 0.7s cubic-bezier(.22,.68,0,1.2) both, borderPulse 4s ease-in-out 1s infinite;
}
.hero-banner::before { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:linear-gradient(180deg,#c8a96e,#4ecdc4); }
.hero-banner::after  { content:''; position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,rgba(200,169,110,0.35),transparent); animation:scanLine 3.5s linear infinite; pointer-events:none; }
.hero-title { font-family:'Syne',sans-serif !important; font-size:2.8rem; font-weight:800; color:#e8e0d5; line-height:1.1; margin:0 0 8px; letter-spacing:-0.02em; animation:fadeSlideDown 0.8s cubic-bezier(.22,.68,0,1.2) 0.1s both; }
.hero-title span { color:#c8a96e; background:linear-gradient(90deg,#c8a96e,#e8c97e,#c8a96e); background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; animation:gradientShift 3s linear infinite; }
.hero-sub { font-size:0.75rem; letter-spacing:0.2em; text-transform:uppercase; color:rgba(232,224,213,0.4); margin:0; animation:fadeIn 1s ease 0.4s both; }
.shimmer-bar { height:2px; background:linear-gradient(90deg,rgba(200,169,110,0) 0%,rgba(200,169,110,0.6) 40%,rgba(78,205,196,0.6) 60%,rgba(200,169,110,0) 100%); background-size:400px 2px; animation:shimmer 2s linear infinite; margin-bottom:28px; border-radius:1px; }

[data-testid="metric-container"] { background:rgba(255,255,255,0.025) !important; border:1px solid rgba(255,255,255,0.07) !important; border-radius:3px !important; padding:18px 20px !important; animation:scaleIn 0.5s cubic-bezier(.22,.68,0,1.2) both, pulseGlow 4s ease-in-out 2s infinite; transition:transform 0.25s ease,border-color 0.25s ease,box-shadow 0.25s ease !important; }
[data-testid="metric-container"]:hover { transform:translateY(-3px) !important; border-color:rgba(200,169,110,0.35) !important; box-shadow:0 8px 28px rgba(200,169,110,0.1) !important; }
[data-testid="metric-container"] label { font-size:0.62rem !important; letter-spacing:0.15em !important; text-transform:uppercase !important; color:rgba(232,224,213,0.4) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; font-size:2rem !important; font-weight:800 !important; color:#c8a96e !important; animation:countUp 0.6s ease both; }

.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid rgba(255,255,255,0.07) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { font-family:'DM Mono',monospace !important; font-size:0.68rem !important; letter-spacing:0.18em !important; text-transform:uppercase !important; color:rgba(232,224,213,0.35) !important; background:transparent !important; border:none !important; padding:12px 24px !important; border-bottom:2px solid transparent !important; transition:color 0.2s ease !important; }
.stTabs [data-baseweb="tab"]:hover { color:rgba(232,224,213,0.7) !important; }
.stTabs [aria-selected="true"] { color:#c8a96e !important; border-bottom-color:#c8a96e !important; }
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none !important; }

.panel-title { font-family:'Syne',sans-serif !important; font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase; color:rgba(200,169,110,0.75); margin-bottom:20px; display:flex; align-items:center; gap:12px; animation:slideInLeft 0.5s ease both; }
.panel-title::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(200,169,110,0.3),transparent); }

.stSelectbox>div>div { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.12) !important; border-radius:3px !important; color:#e8e0d5 !important; font-family:'DM Mono',monospace !important; font-size:0.82rem !important; transition:border-color 0.2s ease,box-shadow 0.2s ease !important; }
.stSelectbox>div>div:focus-within { border-color:#c8a96e !important; box-shadow:0 0 0 2px rgba(200,169,110,0.15) !important; }

.stDataFrame { border:1px solid rgba(255,255,255,0.07) !important; border-radius:3px !important; animation:fadeSlideUp 0.5s ease both; }
.stDataFrame thead th { background:rgba(200,169,110,0.08) !important; color:#c8a96e !important; font-size:0.68rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; border-bottom:1px solid rgba(200,169,110,0.2) !important; }
.stDataFrame tbody td { color:rgba(232,224,213,0.7) !important; font-size:0.78rem !important; border-color:rgba(255,255,255,0.04) !important; transition:background 0.15s ease !important; }
.stDataFrame tbody tr:hover td { background:rgba(200,169,110,0.05) !important; }

.stDownloadButton button { background:rgba(200,169,110,0.1) !important; border:1px solid rgba(200,169,110,0.3) !important; color:#c8a96e !important; font-family:'DM Mono',monospace !important; font-size:0.72rem !important; letter-spacing:0.12em !important; text-transform:uppercase !important; border-radius:2px !important; padding:8px 20px !important; transition:all 0.25s ease !important; animation:ripple 2.5s ease infinite 1.5s; }
.stDownloadButton button:hover { background:rgba(200,169,110,0.2) !important; border-color:rgba(200,169,110,0.55) !important; transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(200,169,110,0.15) !important; }

.stPlotlyChart { animation:scaleIn 0.6s cubic-bezier(.22,.68,0,1.2) both; border-radius:3px; overflow:hidden; }
::-webkit-scrollbar{width:4px;height:4px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:rgba(200,169,110,0.2);border-radius:2px}
.info-text{font-size:0.72rem;color:rgba(232,224,213,0.25);letter-spacing:0.05em;}
.footer-bar{margin-top:60px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.05);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;animation:fadeIn 1s ease 0.5s both;}
[data-testid="column"]:nth-child(1)>div{animation:slideInLeft  0.55s cubic-bezier(.22,.68,0,1.2) 0.1s both;}
[data-testid="column"]:nth-child(2)>div{animation:slideInRight 0.55s cubic-bezier(.22,.68,0,1.2) 0.2s both;}
[data-testid="column"]:nth-child(3)>div{animation:slideInLeft  0.55s cubic-bezier(.22,.68,0,1.2) 0.3s both;}
[data-testid="column"]:nth-child(4)>div{animation:slideInRight 0.55s cubic-bezier(.22,.68,0,1.2) 0.4s both;}
h2,h3{font-family:'Syne',sans-serif !important;color:#e8e0d5 !important;letter-spacing:-0.01em !important;}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d17", "axes.facecolor": "#0d0d17",
    "axes.edgecolor": (1,1,1,0.08), "axes.labelcolor": (0.91,0.88,0.83,0.5),
    "axes.labelsize": 9, "axes.titlecolor": "#e8e0d5", "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": (1,1,1,0.04), "grid.linewidth": 0.8,
    "xtick.color": (0.91,0.88,0.83,0.35), "xtick.labelsize": 8,
    "ytick.color": (0.91,0.88,0.83,0.35), "ytick.labelsize": 8,
    "text.color": "#e8e0d5", "font.family": "monospace",
})

GOLD   = "#c8a96e"
TEAL   = "#4ecdc4"
CORAL  = "#ff6b6b"
PURPLE = "#9b8af0"
MODEL_COLORS = [GOLD, TEAL, PURPLE]
MODEL_NAMES  = ["Linear Regression", "Random Forest", "SVR"]

# ── Load & Preprocess (SINGLE shared pipeline) ────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df_raw = pd.read_csv("Life-Expectancy-Data-Averaged.csv")
    df = df_raw.copy()
    df = df.fillna(df.mean(numeric_only=True))

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Country", "Region", "Economy_status"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df_raw, df   # df_raw for display, df for ML

df_raw, df_ml = load_and_preprocess()

# ── Train ALL 3 models — one shared split + scaler ────────────────────────────
@st.cache_resource
def train_all(_df_ml):
    X = _df_ml.drop("Life_expectancy", axis=1)
    y = _df_ml["Life_expectancy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR":               SVR(kernel="rbf"),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        yp = mdl.predict(X_test_s)
        results[name] = {
            "model": mdl,
            "y_pred": yp,
            "r2":   r2_score(y_test, yp),
            "mse":  mean_squared_error(y_test, yp),
            "rmse": np.sqrt(mean_squared_error(y_test, yp)),
        }
    return results, scaler, X.columns.tolist(), X_test, y_test

all_results, scaler, feature_cols, X_test, y_test = train_all(df_ml)

# Best model by R²
best_model_name = max(all_results, key=lambda m: all_results[m]["r2"])
best_model      = all_results[best_model_name]["model"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-sub">WHO · Global Health Observatory · Machine Learning Project</p>
    <h1 class="hero-title">🌍 Life Expectancy<br><span>Intelligence Dashboard</span></h1>
</div>
<div class="shimmer-bar"></div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
total_countries = df_raw["Country"].nunique()
global_avg      = df_raw["Life_expectancy"].mean()
highest         = df_raw.loc[df_raw["Life_expectancy"].idxmax()]
lowest          = df_raw.loc[df_raw["Life_expectancy"].idxmin()]
best_r2         = all_results[best_model_name]["r2"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Countries",      f"{total_countries}")
c2.metric("Global Avg LE",  f"{global_avg:.1f} yrs")
c3.metric(f"Best · {highest['Country']}", f"{highest['Life_expectancy']:.1f} yrs")
c4.metric(f"Lowest · {lowest['Country']}", f"{lowest['Life_expectancy']:.1f} yrs")
c5.metric(f"Best Model R²", f"{best_r2:.4f}")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Data Preview",
    "🤖  Prediction",
    "📊  Graphs",
    "🗺️  World Map",
    "⚡  Model Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 · Data Preview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Raw Dataset Preview</p>', unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)
    ca.metric("Total Rows",    len(df_raw))
    cb.metric("Total Columns", len(df_raw.columns))
    cc.metric("Countries",     df_raw["Country"].nunique())

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True, height=320)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Descriptive Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Missing Values</p>', unsafe_allow_html=True)
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Column", "Missing Count"]
    if len(missing):
        st.dataframe(missing, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values in dataset!")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 · Prediction  (uses best model from shared pipeline)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(f'<p class="panel-title">Predict with Best Model — {best_model_name} (R²={best_r2:.4f})</p>',
                unsafe_allow_html=True)

    country_list     = sorted(df_raw["Country"].unique())
    selected_country = st.selectbox("Select Country", country_list)
    country_raw      = df_raw[df_raw["Country"] == selected_country]
    country_ml       = df_ml[df_ml.index.isin(country_raw.index)]

    actual    = country_raw["Life_expectancy"].values
    X_country = country_ml.drop("Life_expectancy", axis=1)[feature_cols]
    X_scaled  = scaler.transform(X_country)
    predicted = best_model.predict(X_scaled)
    error     = abs(predicted.mean() - actual.mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Predicted LE",  f"{predicted.mean():.2f} yrs")
    m2.metric("📌 Actual LE",     f"{actual.mean():.2f} yrs")
    m3.metric("Δ Error",          f"{error:.2f} yrs")
    m4.metric("Model Used",       best_model_name)

    # Gauge bar
    pct = min(predicted.mean() / 100, 1.0)
    st.markdown(f"""
    <div style="margin:20px 0 8px; animation:fadeSlideUp 0.6s ease both;">
        <div style="font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:rgba(232,224,213,0.35);margin-bottom:8px;">
            Predicted Life Expectancy — Gauge
        </div>
        <div style="height:10px;background:rgba(255,255,255,0.05);border-radius:5px;overflow:hidden;">
            <div style="height:100%;width:{pct*100:.1f}%;
                background:linear-gradient(90deg,#c8a96e,#4ecdc4);border-radius:5px;
                transition:width 0.8s ease;box-shadow:0 0 12px rgba(200,169,110,0.4);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.62rem;color:rgba(232,224,213,0.25);margin-top:4px;">
            <span>0</span><span>50 yrs</span><span>100 yrs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # All 3 models predict same country
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">All 3 Models — Predictions for this Country</p>',
                unsafe_allow_html=True)
    pred_html = ""
    for i, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        mdl  = all_results[name]["model"]
        pred = mdl.predict(X_scaled).mean()
        diff = abs(pred - actual.mean())
        delay = i * 0.1
        pred_html += f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;
                    animation:slideInLeft 0.5s ease {delay:.1f}s both;">
            <div style="width:160px;font-size:12px;color:rgba(232,224,213,0.6);
                        text-align:right;flex-shrink:0;">{name}</div>
            <div style="flex:1;height:18px;background:rgba(255,255,255,0.04);
                        border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{min(pred/100,1)*100:.1f}%;
                    background:linear-gradient(90deg,{color}99,{color});
                    border-radius:3px;box-shadow:0 0 8px {color}44;
                    animation:slideInLeft 0.9s cubic-bezier(.22,.68,0,1.2) {delay:.1f}s both;">
                </div>
            </div>
            <div style="width:80px;font-size:13px;color:{color};font-weight:700;flex-shrink:0;">
                {pred:.1f} yrs
            </div>
            <div style="font-size:11px;color:rgba(232,224,213,0.3);">err: {diff:.2f}</div>
        </div>"""
    st.markdown(f"<div style='padding:4px 0'>{pred_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Country Health Indicators</p>', unsafe_allow_html=True)
    st.dataframe(country_raw, use_container_width=True)

    report_df = country_raw.copy()
    report_df["Predicted_LE_BestModel"] = predicted
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄  Download Country Report (CSV)",
        data=csv_bytes,
        file_name=f"{selected_country}_life_expectancy_report.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 · Graphs
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="panel-title">Life Expectancy Distribution</p>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df_raw["Life_expectancy"], kde=True, ax=ax1, color=GOLD, alpha=0.6,
                     line_kws={"color": TEAL, "linewidth": 2})
        ax1.axvline(df_raw["Life_expectancy"].mean(), color=CORAL, linestyle="--",
                    linewidth=1.5, label=f"Mean: {df_raw['Life_expectancy'].mean():.1f}")
        ax1.set_xlabel("Life Expectancy (years)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Life Expectancy")
        ax1.legend(fontsize=8)
        fig1.tight_layout(); st.pyplot(fig1)

    with col_r:
        st.markdown('<p class="panel-title">Correlation Heatmap</p>', unsafe_allow_html=True)
        fig_hm, ax_hm = plt.subplots(figsize=(6, 3.5))
        corr = df_ml.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax_hm, cmap="RdYlGn", center=0,
                    linewidths=0.3, linecolor=(1,1,1,0.05),
                    cbar_kws={"shrink": 0.7})
        ax_hm.set_title("Feature Correlation")
        ax_hm.tick_params(labelsize=6)
        fig_hm.tight_layout(); st.pyplot(fig_hm)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<p class="panel-title">GDP per Capita vs Life Expectancy</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sc = ax2.scatter(df_raw["GDP_per_capita"], df_raw["Life_expectancy"],
                         c=df_raw["Life_expectancy"], cmap="YlOrRd",
                         s=50, alpha=0.75, linewidths=0.3, edgecolors="white")
        plt.colorbar(sc, ax=ax2, shrink=0.8)
        ax2.set_xlabel("GDP per Capita (USD)")
        ax2.set_ylabel("Life Expectancy (years)")
        ax2.set_title("GDP vs Life Expectancy")
        fig2.tight_layout(); st.pyplot(fig2)

    with col_r2:
        st.markdown('<p class="panel-title">Schooling vs Life Expectancy</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.scatter(df_raw["Schooling"], df_raw["Life_expectancy"],
                    color=TEAL, alpha=0.65, s=50, linewidths=0.3, edgecolors="white")
        z  = np.polyfit(df_raw["Schooling"].dropna(),
                        df_raw.loc[df_raw["Schooling"].notna(), "Life_expectancy"], 1)
        xs = np.linspace(df_raw["Schooling"].min(), df_raw["Schooling"].max(), 100)
        ax3.plot(xs, np.poly1d(z)(xs), color=GOLD, linewidth=2, linestyle="--", alpha=0.8)
        ax3.set_xlabel("Years of Schooling")
        ax3.set_ylabel("Life Expectancy (years)")
        ax3.set_title("Schooling vs Life Expectancy")
        fig3.tight_layout(); st.pyplot(fig3)

    # Feature importance from RF
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Random Forest — Top 10 Feature Importances</p>',
                unsafe_allow_html=True)
    rf_model  = all_results["Random Forest"]["model"]
    feat_imp  = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    bars4 = ax4.bar(feat_imp.index, feat_imp.values,
                    color=[GOLD,TEAL,PURPLE,CORAL,"#a8e6cf","#ffd93d","#6bcb77","#ff6b9d","#c77dff","#4cc9f0"][:len(feat_imp)],
                    alpha=0.85, width=0.6)
    for bar, val in zip(bars4, feat_imp.values):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                 color=(0.91,0.88,0.83,0.6))
    ax4.set_title("Feature Importance — Random Forest")
    ax4.set_ylabel("Importance Score")
    plt.xticks(rotation=35, ha="right")
    fig4.tight_layout(); st.pyplot(fig4)

    # Animated HTML bars
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    colors_list = [GOLD,TEAL,PURPLE,CORAL,"#a8e6cf","#ffd93d","#6bcb77","#ff6b9d","#c77dff","#4cc9f0"]
    max_imp   = feat_imp.max()
    bars_html = ""
    for i, (feat, val) in enumerate(feat_imp.items()):
        pct   = (val / max_imp) * 100
        color = colors_list[i % len(colors_list)]
        delay = i * 0.08
        bars_html += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;
                    animation:slideInLeft 0.5s ease {delay:.2f}s both;">
            <div style="width:160px;font-size:11px;color:rgba(232,224,213,0.55);
                        text-align:right;flex-shrink:0;overflow:hidden;
                        text-overflow:ellipsis;white-space:nowrap;">{feat}</div>
            <div style="flex:1;height:14px;background:rgba(255,255,255,0.04);
                        border-radius:2px;overflow:hidden;">
                <div style="height:100%;width:{pct:.1f}%;
                    background:linear-gradient(90deg,{color}cc,{color});
                    border-radius:2px;box-shadow:0 0 8px {color}44;
                    animation:slideInLeft 0.9s cubic-bezier(.22,.68,0,1.2) {delay:.2f}s both;"></div>
            </div>
            <div style="width:44px;font-size:11px;color:{color};flex-shrink:0;font-weight:600;">{val:.3f}</div>
        </div>"""
    st.markdown(f"<div style='padding:4px 0'>{bars_html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 · World Map
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Average Life Expectancy by Country</p>', unsafe_allow_html=True)

    world_df = df_raw.groupby("Country")["Life_expectancy"].mean().reset_index()
    fig_map  = px.choropleth(
        world_df, locations="Country", locationmode="country names",
        color="Life_expectancy",
        color_continuous_scale=[[0,"#2d1b00"],[0.3,"#8b4513"],[0.6,"#c8a96e"],[0.8,"#4ecdc4"],[1.0,"#e8e0d5"]],
        labels={"Life_expectancy": "Life Expectancy (yrs)"},
        hover_name="Country", hover_data={"Life_expectancy": ":.1f"},
    )
    fig_map.update_layout(
        paper_bgcolor="#08080f", plot_bgcolor="#08080f",
        geo=dict(bgcolor="#08080f", lakecolor="#0d0d17", landcolor="#1a1a2e",
                 showframe=False, showcoastlines=True,
                 coastlinecolor="rgba(255,255,255,0.08)",
                 showocean=True, oceancolor="#0d0d17",
                 projection_type="natural earth"),
        coloraxis_colorbar=dict(
            title=dict(text="Life Expectancy<br>(years)", font=dict(color="rgba(232,224,213,0.6)", size=10)),
            tickfont=dict(color="rgba(232,224,213,0.5)", size=9),
            bgcolor="rgba(0,0,0,0.3)", outlinecolor="rgba(200,169,110,0.2)",
            thickness=14, len=0.7),
        margin=dict(l=0,r=0,t=10,b=0), height=540,
        font=dict(family="DM Mono, monospace", color="rgba(232,224,213,0.6)"),
    )
    fig_map.update_traces(marker_line_color="rgba(255,255,255,0.05)", marker_line_width=0.5)
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Global Average", f"{world_df['Life_expectancy'].mean():.1f} yrs")
    cs2.metric("Median",         f"{world_df['Life_expectancy'].median():.1f} yrs")
    cs3.metric("Std Deviation",  f"{world_df['Life_expectancy'].std():.1f} yrs")
    cs4.metric("Range",          f"{world_df['Life_expectancy'].max()-world_df['Life_expectancy'].min():.1f} yrs")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 · Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">3 Models — Same Pipeline · Same Split · Same Scaler</p>',
                unsafe_allow_html=True)

    r2_scores   = [all_results[m]["r2"]   for m in MODEL_NAMES]
    mse_scores  = [all_results[m]["mse"]  for m in MODEL_NAMES]
    rmse_scores = [all_results[m]["rmse"] for m in MODEL_NAMES]

    # KPI cards
    k1, k2, k3 = st.columns(3)
    for col, name, color in zip([k1, k2, k3], MODEL_NAMES, MODEL_COLORS):
        r2   = all_results[name]["r2"]
        mse  = all_results[name]["mse"]
        rmse = all_results[name]["rmse"]
        is_best = name == best_model_name
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.025);border:1px solid {color}{'55' if is_best else '22'};
             border-radius:3px;padding:20px 22px;position:relative;overflow:hidden;
             animation:scaleIn 0.5s ease both;{'box-shadow:0 0 20px '+color+'22;' if is_best else ''}">
            <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:{color};"></div>
            <div style="font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;
                        color:{color}99;margin-bottom:10px;">{name} {'🏆' if is_best else ''}</div>
            <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                        color:{color};line-height:1;">{r2:.4f}</div>
            <div style="font-size:0.65rem;color:rgba(232,224,213,0.35);margin-top:4px;">R² Score</div>
            <div style="margin-top:14px;display:flex;gap:20px;">
                <div>
                    <div style="font-size:0.6rem;color:rgba(232,224,213,0.3);letter-spacing:0.12em;text-transform:uppercase;">MSE</div>
                    <div style="font-size:0.9rem;color:rgba(232,224,213,0.7);">{mse:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.6rem;color:rgba(232,224,213,0.3);letter-spacing:0.12em;text-transform:uppercase;">RMSE</div>
                    <div style="font-size:0.9rem;color:rgba(232,224,213,0.7);">{rmse:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Animated R² bars
    st.markdown('<p class="panel-title">R² Score Comparison</p>', unsafe_allow_html=True)
    best_r2_val = max(r2_scores)
    bars_html   = ""
    for i, (name, r2, color) in enumerate(zip(MODEL_NAMES, r2_scores, MODEL_COLORS)):
        pct   = (r2 / best_r2_val) * 100
        delay = i * 0.15
        bars_html += f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;
                    animation:slideInLeft 0.6s ease {delay:.2f}s both;">
            <div style="width:160px;font-size:12px;color:rgba(232,224,213,0.6);
                        text-align:right;flex-shrink:0;">{name}</div>
            <div style="flex:1;height:20px;background:rgba(255,255,255,0.04);
                        border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct:.1f}%;
                    background:linear-gradient(90deg,{color}aa,{color});
                    border-radius:3px;animation:slideInLeft 1s cubic-bezier(.22,.68,0,1.2) {delay:.2f}s both;
                    box-shadow:0 0 10px {color}44;"></div>
            </div>
            <div style="width:80px;font-size:13px;color:{color};font-weight:700;flex-shrink:0;">
                {r2:.4f} {"🏆" if r2==best_r2_val else ""}
            </div>
        </div>"""
    st.markdown(f"<div style='padding:8px 0'>{bars_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<p class="panel-title">R² Score — Bar Chart</p>', unsafe_allow_html=True)
        fig_r2, ax_r2 = plt.subplots(figsize=(6, 3.5))
        bars_r2 = ax_r2.bar(MODEL_NAMES, r2_scores, color=MODEL_COLORS, alpha=0.85, width=0.5)
        for bar, val in zip(bars_r2, r2_scores):
            ax_r2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                       color=(0.91,0.88,0.83,0.8))
        ax_r2.set_ylim(0, 1.1); ax_r2.set_title("R² Score by Model"); ax_r2.set_ylabel("R²")
        ax_r2.axhline(1.0, color=GOLD, linestyle="--", linewidth=0.8, alpha=0.4)
        fig_r2.tight_layout(); st.pyplot(fig_r2)

    with cr:
        st.markdown('<p class="panel-title">RMSE — Bar Chart</p>', unsafe_allow_html=True)
        fig_rm, ax_rm = plt.subplots(figsize=(6, 3.5))
        bars_rm = ax_rm.bar(MODEL_NAMES, rmse_scores, color=MODEL_COLORS, alpha=0.85, width=0.5)
        for bar, val in zip(bars_rm, rmse_scores):
            ax_rm.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                       color=(0.91,0.88,0.83,0.8))
        ax_rm.set_title("RMSE by Model (lower = better)"); ax_rm.set_ylabel("RMSE")
        fig_rm.tight_layout(); st.pyplot(fig_rm)

    # Actual vs Predicted — all 3
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Actual vs Predicted — All Models</p>', unsafe_allow_html=True)
    fig_avp, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name, color in zip(axes, MODEL_NAMES, MODEL_COLORS):
        yp = all_results[name]["y_pred"]
        r2 = all_results[name]["r2"]
        ax.scatter(y_test, yp, color=color, alpha=0.6, s=35, linewidths=0.3, edgecolors="white")
        mn = min(y_test.min(), yp.min()); mx = max(y_test.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], color=CORAL, linewidth=1.5, linestyle="--")
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
    fig_avp.tight_layout(); st.pyplot(fig_avp)

    # Residuals — all 3
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Residual Plots — All Models</p>', unsafe_allow_html=True)
    fig_res, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name, color in zip(axes2, MODEL_NAMES, MODEL_COLORS):
        yp  = all_results[name]["y_pred"]
        res = y_test.values - yp
        ax.scatter(yp, res, color=color, alpha=0.6, s=35, linewidths=0.3, edgecolors="white")
        ax.axhline(0, color=CORAL, linewidth=1.5, linestyle="--")
        ax.set_title(f"{name} — Residuals", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Residual", fontsize=8)
    fig_res.tight_layout(); st.pyplot(fig_res)

    # Summary table
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Summary Table</p>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Model":    MODEL_NAMES,
        "R² Score": [f"{all_results[m]['r2']:.4f}"  for m in MODEL_NAMES],
        "MSE":      [f"{all_results[m]['mse']:.3f}"  for m in MODEL_NAMES],
        "RMSE":     [f"{all_results[m]['rmse']:.3f}" for m in MODEL_NAMES],
        "Best?":    ["🏆" if m == best_model_name else "—" for m in MODEL_NAMES],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <span class="info-text">LIFE EXPECTANCY DASHBOARD · WHO GLOBAL HEALTH OBSERVATORY</span>
    <span class="info-text">LINEAR REGRESSION · RANDOM FOREST · SVR · SCIKIT-LEARN</span>
</div>
""", unsafe_allow_html=True)