"""
Walmart Sales Intelligence Dashboard (advanced)

Run locally:
    streamlit run walmart_dashboard_advanced.py

Requirements:
    pip install streamlit pandas numpy plotly scikit-learn

Place processed_walmart_sales.csv in the data/ folder (or upload it in the app).
"""

import calendar
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# PAGE CONFIG, THEME, CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Walmart Sales Intelligence", page_icon="📊", layout="wide")

BLUE, RED, GREEN, SLATE, AMBER = "#2563eb", "#dc2626", "#16a34a", "#64748b", "#d97706"
PLOT_TEMPLATE = "plotly_white"

st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#1e3a8a 0%,#2563eb 100%); color:#fff;
       padding:22px 28px; border-radius:16px; margin-bottom:18px;}
.hero h1 {color:#fff; margin:0; font-size:2rem;}
.hero p {margin:6px 0 0 0; opacity:.92;}
.brief {background:#fff; border:1px solid #e2e8f0; border-left:5px solid #2563eb;
        border-radius:12px; padding:16px 18px; box-shadow:0 2px 8px rgba(15,23,42,.05);
        height:100%; margin-bottom:12px;}
.brief .t {font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:#64748b; font-weight:600;}
.brief .v {font-size:1.45rem; font-weight:700; color:#0f172a; margin:4px 0 6px 0;}
.brief .x {font-size:.9rem; color:#334155; line-height:1.45;}
.panel {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 18px;
        box-shadow:0 2px 8px rgba(15,23,42,.05); margin-bottom:12px;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; padding:14px 16px;
        border-radius:12px; box-shadow:0 2px 8px rgba(15,23,42,.05);}
.muted {color:#64748b; font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_PATHS = [
    os.path.join("data", "processed_walmart_sales.csv"),
    "processed_walmart_sales.csv",
]
FEATURES = [
    "Dept", "Size", "Store_Type_Code", "IsHoliday", "Month",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
]
ECON = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
FEAT_LABELS = {
    "Dept": "Department", "Size": "Store size", "Store_Type_Code": "Store type",
    "IsHoliday": "Holiday week", "Month": "Month of year", "Temperature": "Temperature",
    "Fuel_Price": "Fuel price", "CPI": "Consumer Price Index", "Unemployment": "Unemployment rate",
}
MONTH_ORDER = [calendar.month_abbr[m] for m in range(1, 13)]

REPORT_FIGS = []  # (title, figure) pairs collected during this run for the HTML report


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def money(x):
    if x is None or pd.isna(x):
        return "–"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def style(fig, height=380, title=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        title=dict(text=title, x=0.01) if title else None,
    )
    return fig


def show(fig, key, title=None, report=True):
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
    )
    if report:
        REPORT_FIGS.append((title or key, fig))


def hist_fig(series, title, bins=40):
    s = series.dropna()
    s = s.clip(upper=s.quantile(0.995))
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(
        x=centers, y=counts, marker_color=BLUE,
        width=(edges[1] - edges[0]) * 0.9,
        hovertemplate="%{x:,.2f}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title=title)
    fig.update_yaxes(title="Count")
    return style(fig, 300, title)


def card(title, value, text, color=BLUE):
    return (
        f"<div class='brief' style='border-left-color:{color}'>"
        f"<div class='t'>{title}</div>"
        f"<div class='v'>{value}</div>"
        f"<div class='x'>{text}</div></div>"
    )


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sales data…")
def load_data(source):
    raw = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Store"] = raw["Store"].astype(int)
    raw["Dept"] = raw["Dept"].astype(int)
    raw["Year"] = raw["Date"].dt.year
    raw["Month"] = raw["Date"].dt.month
    raw["Holiday_Status"] = raw["IsHoliday"].map({1: "Holiday", 0: "Regular day"})
    codes = {t: i for i, t in enumerate(sorted(raw["Store_Type"].unique()))}
    raw["Store_Type_Code"] = raw["Store_Type"].map(codes).astype(int)
    return raw


source = next((p for p in DATA_PATHS if os.path.exists(p)), None)
if source is None:
    st.info("📂 Upload `processed_walmart_sales.csv` to start.")
    up = st.file_uploader("Sales CSV", type="csv")
    if up is None:
        st.stop()
    source = up.getvalue()

df = load_data(source)


# ---------------------------------------------------------------------------
# PREDICTIVE MODEL (trained on all history, validated on the most recent period)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Random Forest model…")
def train_model(_df, n_rows):
    dates = np.sort(_df["Date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.85)])
    train = _df[_df["Date"] < cutoff]
    test = _df[_df["Date"] >= cutoff]

    train_s = train.sample(n=min(100_000, len(train)), random_state=42)
    test_s = test.sample(n=min(60_000, len(test)), random_state=42)

    model = RandomForestRegressor(
        n_estimators=60, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42
    )
    model.fit(train_s[FEATURES], train_s["Weekly_Sales"])

    y_true = test_s["Weekly_Sales"].to_numpy()
    y_pred = model.predict(test_s[FEATURES])

    dept_mean = train_s.groupby("Dept")["Weekly_Sales"].mean()
    baseline = test_s["Dept"].map(dept_mean).fillna(train_s["Weekly_Sales"].mean()).to_numpy()

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "baseline_mae": mean_absolute_error(y_true, baseline),
        "cutoff": cutoff,
        "n_train": len(train_s),
        "n_test": len(test_s),
    }
    test_out = test_s.assign(Predicted=y_pred)[["Date", "Store", "Dept", "Weekly_Sales", "Predicted"]]
    return model, metrics, test_out


model, metrics, test_sample = train_model(df, len(df))


# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🎛️ Filters")
store_types = sorted(df["Store_Type"].unique())
sel_types = st.sidebar.multiselect("Store type", store_types, default=store_types)
sel_depts = st.sidebar.multiselect("Departments (empty = all)", sorted(df["Dept"].unique()), default=[])

min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, format="DD-MM-YYYY"
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) and date_range else min_d

day_type = st.sidebar.radio("Day type", ["All days", "Holiday weeks only", "Regular days only"])

mask = df["Date"].between(pd.Timestamp(start), pd.Timestamp(end)) & df["Store_Type"].isin(sel_types)
if sel_depts:
    mask &= df["Dept"].isin(sel_depts)
if day_type == "Holiday weeks only":
    mask &= df["IsHoliday"] == 1
elif day_type == "Regular days only":
    mask &= df["IsHoliday"] == 0

fdf = df.loc[mask]
if fdf.empty:
    st.warning("No data for the selected filters. Widen the filters to see results.")
    st.stop()

# ---------------------------------------------------------------------------
# SHARED AGGREGATES
# ---------------------------------------------------------------------------
weekly = fdf.groupby("Date", as_index=False).agg(
    Weekly_Sales=("Weekly_Sales", "sum"), IsHoliday=("IsHoliday", "max")
)
weekly["Rolling_4W"] = weekly["Weekly_Sales"].rolling(4, min_periods=1).mean()

store_perf = (
    fdf.groupby(["Store", "Store_Type"])
    .agg(total=("Weekly_Sales", "sum"), avg=("Weekly_Sales", "mean"))
    .reset_index()
    .sort_values("total", ascending=False)
    .reset_index(drop=True)
)

dept_perf = (
    fdf.groupby("Dept")
    .agg(total=("Weekly_Sales", "sum"), avg=("Weekly_Sales", "mean"),
         std=("Weekly_Sales", "std"), rows=("Weekly_Sales", "size"))
    .reset_index()
)
dept_perf["cv"] = dept_perf["std"] / dept_perf["avg"].replace(0, np.nan)
dept_perf = dept_perf.sort_values("total", ascending=False).reset_index(drop=True)

store_type_perf = fdf.groupby("Store_Type", as_index=False)["Weekly_Sales"].sum()


def build_insights(fdf, weekly, store_perf, dept_perf):
    """Turn the filtered data into plain-English briefs."""
    out = []
    total = fdf["Weekly_Sales"].sum()

    out.append(dict(
        title="Revenue scale", value=money(total), color=BLUE,
        text=f"This view covers {len(weekly)} weeks with average weekly revenue of "
             f"{money(weekly['Weekly_Sales'].mean())}.",
    ))

    top, low = store_perf.iloc[0], store_perf.iloc[-1]
    out.append(dict(
        title="Top store", value=f"Store {int(top.Store)}", color=GREEN,
        text=f"Store {int(top.Store)} (type {top.Store_Type}) delivers {top.total / total:.1%} of revenue. "
             f"The weakest store in view, Store {int(low.Store)}, contributes {low.total / total:.1%}.",
    ))

    td = dept_perf.iloc[0]
    out.append(dict(
        title="Leading department", value=f"Dept {int(td.Dept)}", color=BLUE,
        text=f"Department {int(td.Dept)} is the largest revenue line at {money(td.total)} "
             f"({td.total / total:.1%} of sales).",
    ))

    hol = fdf.loc[fdf["IsHoliday"] == 1, "Weekly_Sales"].mean()
    reg = fdf.loc[fdf["IsHoliday"] == 0, "Weekly_Sales"].mean()
    if pd.notna(hol) and pd.notna(reg) and reg:
        lift = hol / reg - 1
        out.append(dict(
            title="Holiday effect", value=f"{lift:+.1%}", lift=lift, color=GREEN if lift >= 0 else RED,
            text=f"Holiday weeks average {money(hol)} per store-department versus {money(reg)} "
                 f"on regular weeks.",
        ))

    mo = weekly.groupby(weekly["Date"].dt.month)["Weekly_Sales"].mean()
    peak, soft = int(mo.idxmax()), int(mo.idxmin())
    out.append(dict(
        title="Seasonality", value=calendar.month_name[peak], color=AMBER,
        text=f"{calendar.month_name[peak]} is the strongest month on average "
             f"({money(mo.max())} per week). {calendar.month_name[soft]} is the softest "
             f"({money(mo.min())}).",
    ))

    yr = weekly.assign(Y=weekly["Date"].dt.year)
    years = sorted(yr["Y"].unique())
    if len(years) >= 2:
        cur, prev = years[-1], years[-2]
        cur_months = set(yr.loc[yr["Y"] == cur, "Date"].dt.month)
        cur_avg = yr.loc[yr["Y"] == cur, "Weekly_Sales"].mean()
        prev_avg = yr.loc[(yr["Y"] == prev) & yr["Date"].dt.month.isin(cur_months), "Weekly_Sales"].mean()
        if prev_avg:
            g = cur_avg / prev_avg - 1
            out.append(dict(
                title=f"Growth {cur} vs {prev}", value=f"{g:+.1%}", color=GREEN if g >= 0 else RED,
                text=f"Like-for-like comparison using only the months present in {cur} "
                     f"(avoids bias from partial years).",
            ))

    vol = dept_perf[dept_perf["rows"] >= 50].sort_values("cv", ascending=False)
    if not vol.empty:
        v = vol.iloc[0]
        out.append(dict(
            title="Highest volatility", value=f"Dept {int(v.Dept)}", color=RED,
            text=f"Week-to-week sales for Department {int(v.Dept)} vary the most "
                 f"(coefficient of variation {v.cv:.2f}), so forecasts for it carry more risk.",
        ))

    samp = fdf.sample(min(100_000, len(fdf)), random_state=0)
    corrs = samp[ECON + ["Weekly_Sales"]].corr()["Weekly_Sales"][ECON]
    k = corrs.abs().idxmax()
    strength = "weak" if abs(corrs[k]) < 0.2 else "moderate" if abs(corrs[k]) < 0.5 else "strong"
    out.append(dict(
        title="Strongest economic link", value=FEAT_LABELS[k], color=SLATE,
        text=f"{FEAT_LABELS[k]} has a {strength} linear relationship with weekly sales "
             f"(r = {corrs[k]:+.2f}).",
    ))
    return out


insights = build_insights(fdf, weekly, store_perf, dept_perf)


# ---------------------------------------------------------------------------
# HEADER & KPIs
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>📊 Walmart Sales Intelligence</h1>"
    "<p>Interactive performance analytics, executive briefs, and an explainable sales forecast.</p></div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total revenue", money(fdf["Weekly_Sales"].sum()))
k2.metric("Avg weekly revenue", money(weekly["Weekly_Sales"].mean()))
k3.metric("Active stores", fdf["Store"].nunique())
k4.metric("Active departments", fdf["Dept"].nunique())
hol_row = next((i for i in insights if i["title"] == "Holiday effect"), None)
k5.metric("Holiday uplift", hol_row["value"] if hol_row else "–")

tab_brief, tab_dash, tab_pred, tab_data, tab_report = st.tabs(
    ["🧭 Executive Brief", "📊 Dashboard", "🔮 Predictive Analysis", "🗂️ Data Explorer", "📝 Summary Report"]
)

# ===========================================================================
# TAB 1 — EXECUTIVE BRIEF
# ===========================================================================
with tab_brief:
    st.subheader("Key findings for the selected view")
    cols = st.columns(3)
    for i, item in enumerate(insights):
        with cols[i % 3]:
            st.markdown(card(item["title"], item["value"], item["text"], item["color"]), unsafe_allow_html=True)

    st.subheader("Recommended actions")
    actions = []
    peak_ins = next(i for i in insights if i["title"] == "Seasonality")
    actions.append(f"Plan inventory and staffing ahead of **{peak_ins['value']}**, the strongest month on average.")
    if hol_row:
        if hol_row["lift"] >= 0:
            actions.append(f"Holiday weeks lift sales (**{hol_row['value']}**). Schedule promotions around holiday calendars.")
        else:
            actions.append(f"Holiday weeks underperform (**{hol_row['value']}**). Review holiday pricing and promotions.")
    vol_ins = next((i for i in insights if i["title"] == "Highest volatility"), None)
    if vol_ins:
        actions.append(f"Add safety stock or tighter monitoring for **{vol_ins['value']}**, the most volatile department.")
    actions.append(f"Investigate **Store {int(store_perf.iloc[-1].Store)}**, the lowest-revenue store in this view.")
    for a in actions:
        st.markdown(f"- {a}")

# ===========================================================================
# TAB 2 — DASHBOARD
# ===========================================================================
with tab_dash:
    # Revenue trend
    st.subheader("Revenue trend")
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=weekly["Date"], y=weekly["Weekly_Sales"], mode="lines", name="Weekly total",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    trend_fig.add_trace(go.Scatter(
        x=weekly["Date"], y=weekly["Rolling_4W"], mode="lines", name="4-week average",
        line=dict(color=SLATE, width=2, dash="dot"),
    ))
    hw = weekly[weekly["IsHoliday"] == 1]
    trend_fig.add_trace(go.Scatter(
        x=hw["Date"], y=hw["Weekly_Sales"], mode="markers", name="Holiday week",
        marker=dict(color=RED, size=9, symbol="diamond"),
    ))
    trend_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(buttons=[
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )
    show(style(trend_fig, 440), "trend", "Revenue trend")

    # Row 2: store-type share, top departments
    c1, c2 = st.columns([1, 2])
    with c1:
        pie = px.pie(store_type_perf, names="Store_Type", values="Weekly_Sales", hole=0.55,
                     color_discrete_sequence=px.colors.qualitative.Bold, title="Revenue by store type")
        pie.update_traces(textinfo="percent+label")
        show(style(pie, 380), "store_type", "Revenue by store type")
    with c2:
        top_d = dept_perf.nlargest(10, "total").sort_values("total").copy()
        top_d["Dept"] = top_d["Dept"].astype(str)
        bar = px.bar(top_d, x="total", y="Dept", orientation="h", color="total",
                     color_continuous_scale="Blues", labels={"total": "Total revenue", "Dept": "Department"},
                     title="Top 10 departments by revenue")
        bar.update_coloraxes(showscale=False)
        show(style(bar, 380), "top_dept", "Top 10 departments")

    # Row 3: store ranking
    st.subheader("Store ranking")
    sp = store_perf.sort_values("total").copy()
    sp["Label"] = "Store " + sp["Store"].astype(str)
    rank = px.bar(sp, x="total", y="Label", orientation="h", color="Store_Type",
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  hover_data={"avg": ":,.0f"}, labels={"total": "Total revenue", "Label": ""})
    show(style(rank, max(420, 18 * len(sp))), "store_rank", "Store ranking")

    # Row 4: holidays + seasonality
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Holiday vs regular weeks (distribution)**")
        samp_h = fdf.sample(min(30_000, len(fdf)), random_state=1)
        box = go.Figure()
        for status, color in [("Regular day", SLATE), ("Holiday", RED)]:
            vals = samp_h.loc[samp_h["Holiday_Status"] == status, "Weekly_Sales"]
            box.add_trace(go.Box(y=vals, name=status, marker_color=color, boxpoints=False))
        box.update_yaxes(title="Weekly sales per store-dept", range=[0, samp_h["Weekly_Sales"].quantile(0.99)])
        show(style(box, 380), "holiday_box", "Holiday vs regular weeks")
    with c4:
        st.markdown("**Monthly seasonality by year**")
        season = fdf.groupby(["Year", "Month"])["Weekly_Sales"].sum().reset_index()
        season["Month_Name"] = season["Month"].map(lambda m: calendar.month_abbr[m])
        line = px.line(season, x="Month_Name", y="Weekly_Sales", color=season["Year"].astype(str),
                       markers=True, category_orders={"Month_Name": MONTH_ORDER},
                       labels={"Weekly_Sales": "Total revenue", "Month_Name": "Month", "color": "Year"})
        show(style(line, 380), "seasonality", "Monthly seasonality by year")

    # Row 5: heatmaps
    st.subheader("Heatmaps")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Avg weekly sales: store type × month**")
        piv_m = (fdf.pivot_table(values="Weekly_Sales", index="Store_Type",
                                 columns="Month", aggfunc="mean")
                 .rename(columns=lambda m: calendar.month_abbr[m]))
        piv_m = piv_m[[c for c in MONTH_ORDER if c in piv_m.columns]]
        show(style(px.imshow(piv_m, text_auto=".2s", color_continuous_scale="Blues", aspect="auto"), 320),
             "heat_month", "Store type × month heatmap")
    with h2:
        st.markdown("**Avg weekly sales: top 15 departments × store type**")
        top15 = dept_perf.nlargest(15, "total")["Dept"]
        piv_d = (fdf[fdf["Dept"].isin(top15)]
                 .pivot_table(values="Weekly_Sales", index="Dept", columns="Store_Type", aggfunc="mean"))
        piv_d.index = piv_d.index.astype(str)
        show(style(px.imshow(piv_d, text_auto=".2s", color_continuous_scale="Blues", aspect="auto"), 320),
             "heat_dept", "Department × store type heatmap")

    # Row 6: correlations
    st.subheader("Correlation matrix")
    corr_cols = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
                 "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    corr_cols = [c for c in corr_cols if c in fdf.columns]
    corr = fdf.sample(min(100_000, len(fdf)), random_state=2)[corr_cols].corr().round(2)
    show(style(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r"), 460),
         "corr", "Correlation matrix")

    # Row 7: distributions
    st.subheader("Distributions")
    d1, d2 = st.columns(2)
    with d1:
        show(hist_fig(fdf["Weekly_Sales"], "Weekly sales (top 0.5% clipped)"), "h_sales", "Weekly sales distribution")
    with d2:
        show(hist_fig(fdf["Temperature"], "Temperature (°F)"), "h_temp", "Temperature distribution")
    d3, d4 = st.columns(2)
    with d3:
        show(hist_fig(fdf["Fuel_Price"], "Fuel price"), "h_fuel", "Fuel price distribution")
    with d4:
        show(hist_fig(fdf["Unemployment"], "Unemployment rate"), "h_unemp", "Unemployment distribution")

# ===========================================================================
# TAB 3 — PREDICTIVE ANALYSIS
# ===========================================================================
with tab_pred:
    st.markdown(
        "This model is a **Random Forest Regressor** trained on store size, department, store type, "
        "month, holiday flag and four economic indicators. It is validated on the **most recent 15% "
        "of weeks**, which the model never saw during training, so the scores reflect real forecasting ability."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (holdout)", f"{metrics['r2']:.3f}", help="Share of sales variation explained on unseen recent weeks.")
    m2.metric("MAE (holdout)", money(metrics["mae"]), help="Average absolute error per store-department-week.")
    m3.metric("RMSE (holdout)", money(metrics["rmse"]), help="Root mean squared error; penalises large misses.")
    improve = 1 - metrics["mae"] / metrics["baseline_mae"]
    m4.metric("Better than dept-average baseline", f"{improve:.0%}",
              help="Error reduction versus simply predicting each department's average sales.")
    st.caption(
        f"Trained on {metrics['n_train']:,} sampled rows before {metrics['cutoff']:%d %b %Y}; "
        f"tested on {metrics['n_test']:,} rows after that date."
    )

    # ---- Scenario builder ----
    st.subheader("🧪 Scenario builder")
    st.caption("Change any input. The forecast and explanation update instantly.")

    s1, s2, s3, s4 = st.columns(4)
    store_sel = s1.selectbox("Store", sorted(df["Store"].unique()), key="p_store")
    dept_sel = s2.selectbox("Department", sorted(df["Dept"].unique()), key="p_dept")
    month_sel = s3.selectbox("Month", list(range(1, 13)), index=10,
                             format_func=lambda m: calendar.month_name[m], key="p_month")
    holiday_sel = s4.toggle("Holiday week", value=False, key="p_hol")

    store_meta = df.loc[df["Store"] == store_sel].iloc[0]
    ctx = df[(df["Store"] == store_sel) & (df["Month"] == month_sel)]
    if ctx.empty:
        ctx = df[df["Store"] == store_sel]

    def lo_hi(col):
        return float(df[col].min()), float(df[col].max())

    def med(col):
        return float(ctx[col].median())

    sk = f"{store_sel}_{month_sel}"
    e1, e2, e3, e4 = st.columns(4)
    t_lo, t_hi = lo_hi("Temperature")
    f_lo, f_hi = lo_hi("Fuel_Price")
    c_lo, c_hi = lo_hi("CPI")
    u_lo, u_hi = lo_hi("Unemployment")
    temp = e1.slider("Temperature (°F)", t_lo, t_hi, med("Temperature"), 0.5, key=f"t_{sk}")
    fuel = e2.slider("Fuel price ($)", f_lo, f_hi, med("Fuel_Price"), 0.01, key=f"f_{sk}")
    cpi = e3.slider("CPI", c_lo, c_hi, med("CPI"), 0.1, key=f"c_{sk}")
    unemp = e4.slider("Unemployment (%)", u_lo, u_hi, med("Unemployment"), 0.01, key=f"u_{sk}")

    scenario = {
        "Dept": dept_sel, "Size": int(store_meta["Size"]),
        "Store_Type_Code": int(store_meta["Store_Type_Code"]),
        "IsHoliday": int(holiday_sel), "Month": month_sel,
        "Temperature": temp, "Fuel_Price": fuel, "CPI": cpi, "Unemployment": unemp,
    }
    ref = {f: float(df[f].median()) for f in FEATURES}

    # One batch: scenario, typical reference, and each input swapped to typical
    rows = [scenario, ref] + [{**scenario, f: ref[f]} for f in FEATURES]
    X_batch = pd.DataFrame(rows)[FEATURES]
    preds = model.predict(X_batch)
    pred, typical_pred = float(preds[0]), float(preds[1])
    alt = preds[2:]

    # Uncertainty band from the spread of individual trees
    X_one = pd.DataFrame([scenario])[FEATURES]
    tree_preds = np.array([t.predict(X_one.to_numpy())[0] for t in model.estimators_])
    lo_band, hi_band = np.percentile(tree_preds, [10, 90])

    impact = pd.DataFrame({
        "Factor": [FEAT_LABELS[f] for f in FEATURES],
        "Impact": pred - alt,
    })

    hist_ref = df[(df["Store"] == store_sel) & (df["Dept"] == dept_sel)]
    hist_same = hist_ref[hist_ref["IsHoliday"] == int(holiday_sel)]
    ref_sales = hist_ref["Weekly_Sales"] if len(hist_ref) >= 10 else df["Weekly_Sales"]
    hist_mean = float(ref_sales.mean())
    q25, q75 = float(ref_sales.quantile(0.25)), float(ref_sales.quantile(0.75))
    gauge_max = max(float(df["Weekly_Sales"].quantile(0.99)), pred * 1.15)

    # ---- Forecast result ----
    st.markdown("---")
    r1, r2 = st.columns([1, 1])
    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={"reference": hist_mean, "valueformat": ",.0f", "prefix": "$"},
            title={"text": f"Predicted weekly sales<br><span style='font-size:0.8em'>"
                           f"Store {store_sel} · Dept {dept_sel} · {calendar.month_name[month_sel]}</span>"},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": BLUE},
                "steps": [
                    {"range": [0, q25], "color": "#fee2e2"},
                    {"range": [q25, q75], "color": "#fef9c3"},
                    {"range": [q75, gauge_max], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": hist_mean},
            },
        ))
        show(style(gauge, 360), "gauge", None, report=False)
    with r2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(f"### {money(pred)} per week")
        st.markdown(f"**Likely range (10th–90th percentile of model trees):** {money(lo_band)} to {money(hi_band)}")
        st.markdown(
            f"**Historical average for this store and department:** {money(hist_mean)} "
            f"(holiday weeks: {money(hist_same['Weekly_Sales'].mean()) if len(hist_same) else 'n/a'})"
        )
        vs_typical = (pred / typical_pred - 1) if typical_pred else np.nan
        st.markdown(f"**Versus a typical week in the data:** {vs_typical:+.1%}")
        top2 = impact.reindex(impact["Impact"].abs().sort_values(ascending=False).index).head(2)
        drivers = " and ".join(
            f"**{r.Factor}** ({'+' if r.Impact >= 0 else '−'}{money(abs(r.Impact))})" for r in top2.itertuples()
        )
        st.markdown(f"**Main drivers of this forecast:** {drivers}.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "**How to read this:** the model predicts the expected weekly sales for one store-department "
        "in the chosen month, given the inputs. The grey band shows how much the individual trees "
        "disagree, a rough guide to uncertainty, not a guarantee."
    )

    # ---- Explanation ----
    st.subheader("🔍 Why this prediction?")
    st.caption(
        "Each bar shows how much the forecast changes when that input is replaced by a typical value "
        "from the data. Green pushes the forecast up, red pushes it down."
    )
    impact_sorted = impact.sort_values("Impact")
    contrib = go.Figure(go.Bar(
        x=impact_sorted["Impact"], y=impact_sorted["Factor"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in impact_sorted["Impact"]],
        text=[f"{v:+,.0f}" for v in impact_sorted["Impact"]], textposition="outside",
        hovertemplate="%{y}: %{x:+,.0f}<extra></extra>",
    ))
    contrib.update_xaxes(title="Change in predicted weekly sales ($)", zeroline=True, zerolinewidth=2)
    show(style(contrib, 380), "contrib", None, report=False)

    # ---- Model transparency ----
    st.subheader("🧠 What drives sales overall?")
    st.caption("Global feature importance across the whole model (share of total explanatory power).")
    imp_df = pd.DataFrame({
        "Feature": [FEAT_LABELS[f] for f in FEATURES],
        "Importance": model.feature_importances_,
    }).sort_values("Importance")
    imp_fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance",
                     color_continuous_scale="Blues", labels={"Importance": "Importance score"})
    imp_fig.update_coloraxes(showscale=False)
    show(style(imp_fig, 380), "importance", "Feature importance")
    top_feat = imp_df.iloc[-1]
    econ_share = imp_df[imp_df["Feature"].isin([FEAT_LABELS[f] for f in ECON])]["Importance"].sum()
    st.markdown(
        f"<div class='panel'>The most influential input overall is <b>{top_feat['Feature']}</b>. "
        f"The four economic indicators together account for <b>{econ_share:.0%}</b> of the model's "
        f"importance. Store and department mix matter more than the economy, so use the forecast for "
        f"planning ranges rather than exact figures.</div>",
        unsafe_allow_html=True,
    )

    # ---- Accuracy diagnostics ----
    st.subheader("🎯 Model accuracy on recent, unseen weeks")
    diag = test_sample.sample(min(3000, len(test_sample)), random_state=3)
    lim = float(max(diag["Weekly_Sales"].quantile(0.99), diag["Predicted"].quantile(0.99)))
    acc = go.Figure()
    acc.add_trace(go.Scattergl(
        x=diag["Weekly_Sales"], y=diag["Predicted"], mode="markers",
        marker=dict(color=BLUE, size=5, opacity=0.35), name="Test rows",
        hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
    ))
    acc.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                             line=dict(color=RED, dash="dash")))
    acc.update_xaxes(title="Actual weekly sales", range=[0, lim])
    acc.update_yaxes(title="Predicted weekly sales", range=[0, lim])
    show(style(acc, 420), "accuracy", "Actual vs predicted (holdout)")

# ===========================================================================
# TAB 4 — DATA EXPLORER
# ===========================================================================
with tab_data:
    st.subheader("Filtered records")
    default_cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday", "Store_Type", "Temperature", "CPI", "Unemployment"]
    cols_sel = st.multiselect("Columns", list(fdf.columns), default=[c for c in default_cols if c in fdf.columns])
    if cols_sel:
        st.dataframe(fdf[cols_sel].head(2000), width="stretch", height=420)
        st.caption(f"Showing the first 2,000 of {len(fdf):,} filtered rows.")

    st.subheader("Export aggregated data")
    agg = (fdf.groupby(["Store", "Store_Type", "Dept", "Year", "Month"], as_index=False)
           .agg(Total_Sales=("Weekly_Sales", "sum"), Avg_Weekly_Sales=("Weekly_Sales", "mean")))
    st.download_button(
        "⬇️ Download aggregated CSV",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name="walmart_sales_aggregated.csv",
        mime="text/csv",
    )

# ===========================================================================
# TAB 5 — SUMMARY REPORT (built last so it can use every figure from this run)
# ===========================================================================
def build_html_report(title, kpis, insight_list, figs):
    fig_html = "".join(
        f"<section><h2>{name}</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</section>"
        for name, fig in figs
    )
    kpi_html = "".join(f"<div class='kpi'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in kpis)
    ins_html = "".join(
        f"<div class='card' style='border-left-color:{i['color']}'><div class='t'>{i['title']}</div>"
        f"<div class='v'>{i['value']}</div><p>{i['text']}</p></div>"
        for i in insight_list
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:32px;}}
h1{{margin:0 0 4px 0}} .sub{{color:#64748b;margin-bottom:24px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}}
.kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 18px;min-width:180px}}
.kpi .k{{font-size:12px;color:#64748b;text-transform:uppercase}} .kpi .v{{font-size:22px;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-left:5px solid #2563eb;border-radius:12px;padding:14px 16px}}
.card .t{{font-size:12px;color:#64748b;text-transform:uppercase}} .card .v{{font-size:20px;font-weight:700}}
section{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:18px}}
</style></head>
<body>
<h1>{title}</h1>
<div class="sub">Generated for the filters active at the time of export.</div>
<div class="kpis">{kpi_html}</div>
<h2>Key findings</h2>
<div class="cards">{ins_html}</div>
{fig_html}
</body></html>"""


with tab_report:
    st.subheader("📝 Auto-generated summary report")
    st.markdown("Narrative summary for the current filters. Generate the full HTML report below to share it.")

    narrative = " ".join(f"**{i['title']}:** {i['text']}" for i in insights)
    st.markdown(f"<div class='panel'>{narrative}</div>", unsafe_allow_html=True)

    st.markdown("**Predictive summary**")
    st.markdown(
        f"<div class='panel'>The Random Forest model explains <b>{metrics['r2']:.0%}</b> of the variation in "
        f"weekly sales on unseen recent weeks, with an average error of <b>{money(metrics['mae'])}</b> per "
        f"store-department-week. It beats a simple department-average baseline by <b>{improve:.0%}</b>. "
        f"The current scenario forecasts <b>{money(pred)}</b> per week for Store {store_sel}, "
        f"Department {dept_sel}, in {calendar.month_name[month_sel]}.</div>",
        unsafe_allow_html=True,
    )

    kpis = [
        ("Total revenue", money(fdf["Weekly_Sales"].sum())),
        ("Avg weekly revenue", money(weekly["Weekly_Sales"].mean())),
        ("Active stores", str(fdf["Store"].nunique())),
        ("Model R²", f"{metrics['r2']:.3f}"),
    ]
    if st.button("Build HTML report", type="primary"):
        st.session_state["report_html"] = build_html_report(
            "Walmart Sales Intelligence Report", kpis, insights, REPORT_FIGS,
        )
    if "report_html" in st.session_state:
        st.download_button(
            "⬇️ Download HTML report",
            data=st.session_state["report_html"].encode("utf-8"),
            file_name="walmart_sales_report.html",
            mime="text/html",
        )
        st.caption("Rebuild the report after changing filters to refresh its contents.")
