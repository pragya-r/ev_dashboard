import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import pydeck as pdk
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="State Level Aggregation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (mirrors generate_bandwidth_charts.py)
# ─────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/ev_bandwidth_results.parquet")
LOOKUP_PATH = Path("utils/postcode_suburb_lookup.csv")
POA_PATH    = Path("utils/postal_areas.geojson")
STATES_PATH = Path("utils/states.geojson")

HOUR_COLS = [f"hour_{h:02d}_mbps" for h in range(24)]
KEY_COLS  = ["year", "state", "scenario"]
REP_CANDIDATES = ("rep", "replication", "rep_id", "run", "run_id")
SCENARIO_COLOURS = {"LOW": "#2196F3", "HIGH": "#F44336"}
LO, HI = 5, 95
YLIM_HEADROOM = 1.35

GRADIENT_STEPS = 12

# Decimal bit-rate units, largest first. Input data are in Mbps.
UNITS = (("Tbps", 1e6), ("Gbps", 1e3), ("Mbps", 1.0))

# Hours by which each state's local standard time is behind AEST (UTC+10).
STATE_LAG_H = {"NSW": 0.0, "VIC": 0.0, "ACT": 0.0, "TAS": 0.0, "QLD": 0.0,
               "SA": 0.5, "NT": 0.5, "WA": 2.0}

# Time-zone label shown on each chart's x-axis ("ALL" = national, in AEST).
TZ_LABEL = {"NSW": "AEST, UTC+10", "VIC": "AEST, UTC+10", "ACT": "AEST, UTC+10",
            "TAS": "AEST, UTC+10", "QLD": "AEST, UTC+10",
            "SA": "ACST, UTC+9:30", "NT": "ACST, UTC+9:30",
            "WA": "AWST, UTC+8", "ALL": "AEST, UTC+10"}

# Abbreviation → full name, matching states.geojson STATE_NAME values
STATE_ABBREV = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "WA":  "Western Australia",
    "SA":  "South Australia",
    "TAS": "Tasmania",
    "ACT": "Australian Capital Territory",
    "NT":  "Northern Territory",
}


# ─────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────
def to_twelve_hour(hour: int) -> str:
    hour = int(hour) % 24
    if hour == 0:
        return "12AM"
    if hour < 12:
        return f"{hour}AM"
    if hour == 12:
        return "12PM"
    return f"{hour - 12}PM"


HOUR_LABELS = [to_twelve_hour(h) for h in range(24)]


def pick_unit(ymax_mbps: float):
    """Largest unit in which the axis maximum is >= 1."""
    for name, div in UNITS:
        if ymax_mbps >= div:
            return name, div
    return "Mbps", 1.0


def fmt_rate(mbps: float, unit) -> str:
    name, div = unit
    x = mbps / div
    digits = 0 if x >= 100 else 1 if x >= 10 else 2
    return f"{x:,.{digits}f} {name}"


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def to_aest(B: np.ndarray, lag: float) -> np.ndarray:
    """Shift (R, 24) hourly profiles forward by `lag` hours, circularly.
    Fractional lag is handled by linear blending of adjacent bins."""
    whole = int(lag // 1)
    frac = lag - whole
    out = np.roll(B, whole, axis=1)
    if frac:
        out = (1 - frac) * out + frac * np.roll(out, 1, axis=1)
    return out


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────
def _norm_postcode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(4)


@st.cache_data(show_spinner="Aggregating replications…")
def load_aggregates():
    """Read the postcode-level Parquet one year at a time and reduce it to:

    pc_means : postcode × year × scenario means across replications (map layer)
    by_state : per-replication sums over postcodes, local standard time
    by_nat   : per-replication national sums, every state shifted to AEST
    ymax     : y-axis upper bound per (group) — largest P95 over all years/scenarios
    n_reps   : number of replications found
    """
    schema_cols = pq.ParquetFile(DATA_PATH).schema.names
    required = set(KEY_COLS) | {"postcode", "ev_count", "total_upstream_gb"} | set(HOUR_COLS)
    missing = required - set(schema_cols)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    rep_col = next((c for c in REP_CANDIDATES if c in schema_cols), None)
    value_cols = HOUR_COLS + ["ev_count", "total_upstream_gb"]
    if "nbn_ev_count" in schema_cols:
        value_cols.append("nbn_ev_count")
    load_cols = list(dict.fromkeys(KEY_COLS + ["postcode"] + value_cols + ([rep_col] if rep_col else [])))

    years = sorted(pd.read_parquet(DATA_PATH, columns=["year"])["year"].unique())

    pc_parts, st_parts = [], []
    for y in years:
        df = pd.read_parquet(DATA_PATH, columns=load_cols, filters=[("year", "==", y)])
        df = df[df["scenario"].isin(SCENARIO_COLOURS)]
        if rep_col is None:
            df["rep"] = 0
        elif rep_col != "rep":
            df = df.rename(columns={rep_col: "rep"})
        df["postcode"] = _norm_postcode(df["postcode"])

        pc_cols = ["ev_count", "total_upstream_gb"] + (["nbn_ev_count"] if "nbn_ev_count" in value_cols else [])
        pc_parts.append(df.groupby(KEY_COLS + ["postcode"], as_index=False)[pc_cols].mean())
        st_parts.append(df.groupby(["state", "year", "scenario", "rep"], as_index=False)[value_cols].sum())
        del df

    pc_means = pd.concat(pc_parts, ignore_index=True)
    by_state = pd.concat(st_parts, ignore_index=True)

    unknown = set(by_state["state"].unique()) - (STATE_LAG_H.keys() & TZ_LABEL.keys())
    if unknown:
        raise ValueError(f"No timezone defined for states: {sorted(unknown)}")

    by_aest = by_state.copy()
    for state, lag in STATE_LAG_H.items():
        if lag:
            m = by_aest["state"] == state
            by_aest.loc[m, HOUR_COLS] = to_aest(by_aest.loc[m, HOUR_COLS].to_numpy(), lag)
    by_nat = by_aest.groupby(["year", "scenario", "rep"], as_index=False)[value_cols].sum()
    del by_aest

    # Shared y-axis per group (state, or "ALL"): largest P95 over every year and scenario
    ymax = {}
    for state, g in by_state.groupby("state"):
        ymax[state] = _group_ymax(g)
    ymax["ALL"] = _group_ymax(by_nat)

    n_reps = int(by_state["rep"].nunique())
    return pc_means, by_state, by_nat, ymax, n_reps


def _group_ymax(g: pd.DataFrame) -> float:
    top = 0.0
    for _, sub in g.groupby(["year", "scenario"]):
        top = max(top, float(np.percentile(sub[HOUR_COLS].to_numpy(), HI, axis=0).max()))
    return top * YLIM_HEADROOM


@st.cache_data(show_spinner=False)
def load_postcode_reps(year, postcode):
    """Per-replication hourly profile of one postcode (both scenarios)."""
    df = pd.read_parquet(
        DATA_PATH,
        columns=[c for c in pq.ParquetFile(DATA_PATH).schema.names
                 if c in set(KEY_COLS + ["postcode", "ev_count", "total_upstream_gb", "nbn_ev_count"]
                             + HOUR_COLS)],
        filters=[("year", "==", year)],
    )
    df["postcode"] = _norm_postcode(df["postcode"])
    return df[df["postcode"] == postcode]


@st.cache_data
def load_suburbs():
    try:
        lk = pd.read_csv(LOOKUP_PATH, dtype=str).rename(columns={"POSTCODE": "postcode"})
        lk["postcode"] = _norm_postcode(lk["postcode"])
        return lk[["postcode", "SUBURBS_JOINED"]].drop_duplicates("postcode")
    except FileNotFoundError:
        return pd.DataFrame(columns=["postcode", "SUBURBS_JOINED"])


@st.cache_data
def load_geodata():
    gdf = gpd.read_file(POA_PATH)
    for col in gdf.columns:
        if col.upper() in ("POA_CODE", "POSTCODE", "POA_CODE21"):
            gdf = gdf.rename(columns={col: "postcode"})
            break
    gdf["postcode"] = _norm_postcode(gdf["postcode"])

    gdf_states = gpd.read_file(STATES_PATH).rename(columns={"STATE_NAME": "state"})
    return gdf[["postcode", "geometry"]], gdf_states


# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
def summarise(sub: pd.DataFrame) -> dict:
    """sub: one row per replication (already summed over postcodes)."""
    B = sub[HOUR_COLS].to_numpy()
    mean_profile = B.mean(axis=0)
    ph = int(mean_profile.argmax())
    return {
        "n_reps":       B.shape[0],
        "mean_profile": mean_profile,
        "lo_profile":   np.percentile(B, LO, axis=0),
        "hi_profile":   np.percentile(B, HI, axis=0),
        "peak_hour":    ph,
        "peak_mbps":    float(mean_profile[ph]),
        "ev_count":     float(sub["ev_count"].mean()),
        "nbn_ev_count": float(sub["nbn_ev_count"].mean()) if "nbn_ev_count" in sub else None,
        "gb_mean":      float(sub["total_upstream_gb"].mean()),
        "gb_lo":        float(np.percentile(sub["total_upstream_gb"], LO)),
        "gb_hi":        float(np.percentile(sub["total_upstream_gb"], HI)),
    }


def summarise_scenarios(sub_all: pd.DataFrame) -> dict:
    out = {}
    for scen in SCENARIO_COLOURS:
        sub = sub_all[sub_all["scenario"] == scen]
        if not sub.empty:
            out[scen] = summarise(sub)
    return out


# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
def _gradient_colors():
    """Red-white gradient: intense red → white (low)."""
    gb = np.linspace(0, 240, GRADIENT_STEPS).astype(int)
    return [[255, int(g), int(g)] for g in gb]


GRADIENT = _gradient_colors()


def value_to_color(value, min_val, max_val, use_log=True, alpha=200):
    if value <= 0:
        return [255, 255, 255, 80]
    v  = np.log1p(value)   if use_log else value
    mn = np.log1p(min_val) if use_log else min_val
    mx = np.log1p(max_val) if use_log else max_val
    norm = np.clip((v - mn) / (mx - mn + 1e-9), 0, 1)
    step = int(norm * (GRADIENT_STEPS - 1))
    step = GRADIENT_STEPS - 1 - step   # flip: high value → deep red
    return GRADIENT[step] + [alpha]


@st.cache_data
def get_global_colour_ranges(_pc_means, metrics):
    """Fixed colour scales across all years/scenarios, per metric and resolution."""
    state_df = _pc_means.groupby(["year", "scenario", "state"], as_index=False)[list(metrics)].sum()
    pc_r = {m: (float(_pc_means[m].min()), float(_pc_means[m].max())) for m in metrics}
    st_r = {m: (float(state_df[m].min()),  float(state_df[m].max()))  for m in metrics}
    return pc_r, st_r


# ─────────────────────────────────────────────────────────────
# MAP BUILDERS  (cached per year / scenario / metric)
# ─────────────────────────────────────────────────────────────
def _display_cols(gdf, metric_cols):
    gdf["EVs_Display"]  = gdf["ev_count"].apply(lambda x: f"{x:,.0f}")
    gdf["Data_Display"] = gdf["total_upstream_gb"].apply(lambda x: f"{x:,.2f} GB")
    if "nbn_ev_count" in metric_cols:
        gdf["NBN_Display"] = gdf["nbn_ev_count"].apply(lambda x: f"{x:,.0f}")


@st.cache_data(show_spinner=False)
def build_postcode_map(year, scenario, metric, colour_range):
    metric_cols = [m for m in ("ev_count", "total_upstream_gb", "nbn_ev_count") if m in PC_MEANS.columns]
    agg = (PC_MEANS[(PC_MEANS["year"] == year) & (PC_MEANS["scenario"] == scenario)]
           [["postcode", "state"] + metric_cols]
           .merge(load_suburbs(), on="postcode", how="left"))
    gdf = GDF_AREAS.merge(agg, on="postcode", how="left")
    gdf[metric_cols] = gdf[metric_cols].fillna(0)
    gdf["SUBURBS_JOINED"] = gdf["SUBURBS_JOINED"].fillna("(Unknown)")
    gdf["state"] = gdf["state"].fillna("(Unknown)")
    mn, mx = colour_range
    gdf["fill_color"] = gdf[metric].apply(lambda v: value_to_color(v, mn, mx))
    gdf["line_color"] = [[180, 180, 180, 160]] * len(gdf)
    _display_cols(gdf, metric_cols)
    return json.loads(gdf.to_json())


@st.cache_data(show_spinner=False)
def build_state_map(year, scenario, metric, colour_range):
    metric_cols = [m for m in ("ev_count", "total_upstream_gb", "nbn_ev_count") if m in PC_MEANS.columns]
    sub = PC_MEANS[(PC_MEANS["year"] == year) & (PC_MEANS["scenario"] == scenario)]
    agg = sub.groupby("state", as_index=False)[metric_cols].sum()
    agg["state"] = agg["state"].map(STATE_ABBREV).fillna(agg["state"])
    gdf = GDF_STATES.merge(agg, on="state", how="left")
    gdf[metric_cols] = gdf[metric_cols].fillna(0)
    mn, mx = colour_range
    gdf["fill_color"] = gdf[metric].apply(lambda v: value_to_color(v, mn, mx))
    gdf["line_color"] = [[120, 120, 120, 200]] * len(gdf)
    _display_cols(gdf, metric_cols)
    return json.loads(gdf.to_json())


# ─────────────────────────────────────────────────────────────
# HOURLY PROFILE CHART  (LOW / HIGH overlay, mean + P5–P95 band)
# ─────────────────────────────────────────────────────────────
def profile_chart(summaries, title, tz_label, ymax, unit):
    name, div = unit
    k = 1.0 / div
    hrs = list(range(24))

    fig = go.Figure()
    for scen, s in summaries.items():
        colour = SCENARIO_COLOURS[scen]
        fig.add_trace(go.Scatter(
            x=hrs, y=s["hi_profile"] * k, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=hrs, y=s["lo_profile"] * k, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_rgba(colour, 0.3),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=hrs, y=s["mean_profile"] * k, mode="lines+markers",
            line=dict(color=colour, width=2), marker=dict(size=6),
            name=f"{scen}: {fmt_rate(s['peak_mbps'], unit)} at {HOUR_LABELS[s['peak_hour']]}",
            hovertemplate=f"{scen}<br>%{{x}}:00 — %{{y:.3g}} {name}<extra></extra>",
        ))
        fig.add_vline(x=s["peak_hour"], line_dash="dot", line_color="black", line_width=1.5)

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="black", dash="dot", width=1.5), name="Peak-hour",
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}: Avg. Aggregate Bandwidth per Hour</b>", font_size=16),
        xaxis=dict(
            title=dict(text=f"<b>Hour of day ({tz_label})</b>"),
            tickmode="array", tickvals=hrs[::3], ticktext=HOUR_LABELS[::3],
            range=[-0.5, 23.5], showgrid=True, gridcolor="rgba(0,0,0,0.15)",
        ),
        yaxis=dict(
            title=dict(text=f"<b>Bandwidth ({name})</b>"),
            range=[0, ymax * k], gridcolor="rgba(0,0,0,0.15)",
        ),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="grey", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=460, margin=dict(t=70, b=60, l=70, r=30),
    )
    return fig


def scenario_caption(summaries):
    lines = []
    for scen, s in summaries.items():
        nbn = f", {s['nbn_ev_count']:,.0f} on NBN" if s["nbn_ev_count"] is not None else ""
        lines.append(
            f"**{scen}** — EVs: {s['ev_count']:,.0f} total{nbn} · "
            f"NBN upload: {s['gb_mean']:,.1f} GB/day (P{LO}–P{HI}: {s['gb_lo']:,.1f}–{s['gb_hi']:,.1f})"
        )
    st.caption("  \n".join(lines))


# ─────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────
try:
    PC_MEANS, BY_STATE, BY_NAT, YMAX, N_REPS = load_aggregates()
    GDF_AREAS, GDF_STATES = load_geodata()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}\n\nExpected `{DATA_PATH}` and the GeoJSON files in `utils`.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

metric_options = ["total_upstream_gb", "ev_count"] + (["nbn_ev_count"] if "nbn_ev_count" in PC_MEANS else [])
METRIC_LABELS = {"total_upstream_gb": "Daily Upload (GB, NBN EVs only)",
                 "ev_count": "Number of EVs (total fleet)",
                 "nbn_ev_count": "Number of NBN-connected EVs"}
NBN_NOTE = ("**Upload volume (GB) and bandwidth (Mbps/Gbps) are from NBN-connected EVs only.** "
            "EV counts are the total fleet unless labelled NBN.")
postcode_ranges, state_ranges = get_global_colour_ranges(PC_MEANS, tuple(metric_options))

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
years     = sorted(PC_MEANS["year"].unique())
scenarios = [s for s in SCENARIO_COLOURS if s in set(PC_MEANS["scenario"])]

selected_year     = st.sidebar.selectbox("Year", options=years, index=0)
selected_scenario = st.sidebar.selectbox("Scenario (map and headline metrics)", options=scenarios, index=0)
metric = st.sidebar.radio("Map Metric", metric_options, format_func=METRIC_LABELS.get)
map_level = st.sidebar.radio("Map Resolution", ["Postcode level", "State level"])

# ─────────────────────────────────────────────────────────────
# HEADER METRICS  (means across replications; national peak in AEST)
# ─────────────────────────────────────────────────────────────
nat_sub  = BY_NAT[(BY_NAT["year"] == selected_year)]
nat_summ = summarise_scenarios(nat_sub)
sel      = nat_summ[selected_scenario]
nat_unit = pick_unit(YMAX["ALL"])
df_year  = PC_MEANS[(PC_MEANS["year"] == selected_year) & (PC_MEANS["scenario"] == selected_scenario)]

st.title("EV Broadband Impact Dashboard", anchor=False)
st.info(NBN_NOTE)
st.caption(
    f"Ensemble of {N_REPS:,} Monte Carlo replications — values are means; bands are P{LO}–P{HI}. "
    f"Scenario: **{selected_scenario}** · Year: **{selected_year}**"
)
kpis = st.columns(5 if sel["nbn_ev_count"] is not None else 4)
kpis[0].metric("Total EVs (fleet)", f"{sel['ev_count']:,.0f}")
i = 1
if sel["nbn_ev_count"] is not None:
    kpis[1].metric("NBN-connected EVs", f"{sel['nbn_ev_count']:,.0f}")
    i = 2
kpis[i].metric("Daily Upload (TB, NBN only)", f"{sel['gb_mean'] / 1000:,.2f}")
kpis[i + 1].metric("Peak Bandwidth (AEST, NBN only)", fmt_rate(sel["peak_mbps"], nat_unit),
                   help=f"Peak of the mean national NBN-only profile, at {HOUR_LABELS[sel['peak_hour']]} AEST.")
kpis[i + 2].metric("Postcodes with EVs", f"{int((df_year['ev_count'] > 0).sum()):,}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
st.subheader(f"Geographic Distribution ({map_level})", anchor=False)
view_state = pdk.ViewState(latitude=-25.5, longitude=134.0, zoom=3, pitch=0)

_nbn_row = ('<b>NBN EVs:</b> {NBN_Display}<br/>' if "nbn_ev_count" in PC_MEANS else "")

tooltip_postcode = {
    "html": f"""
    <div style="font-family:Arial,sans-serif;padding:10px;background:rgba(255,255,255,0.95);
                border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);max-width:280px;">
        <b style="font-size:14px;">Postcode: {{postcode}}</b><br/>
        <span style="color:#666;font-size:12px;">{{state}} — {{SUBURBS_JOINED}}</span>
        <hr style="border:none;border-top:1px solid #ddd;margin:6px 0;">
        <b>EVs (total):</b> {{EVs_Display}}<br/>
        {_nbn_row}
        <b>Daily Upload (NBN EVs):</b> {{Data_Display}}
    </div>""",
    "style": {"backgroundColor": "transparent", "border": "none"},
}

tooltip_state = {
    "html": f"""
    <div style="font-family:Arial,sans-serif;padding:10px;background:rgba(255,255,255,0.95);
                border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size:14px;">{{state}}</b><br/>
        <b>EVs (total):</b> {{EVs_Display}}<br/>
        {_nbn_row}
        <b>Daily Upload (NBN EVs):</b> {{Data_Display}}
    </div>""",
    "style": {"backgroundColor": "transparent", "border": "none"},
}


def render_map():
    if map_level == "Postcode level":
        geo, tip = build_postcode_map(selected_year, selected_scenario, metric,
                                      postcode_ranges[metric]), tooltip_postcode
    else:
        geo, tip = build_state_map(selected_year, selected_scenario, metric,
                                   state_ranges[metric]), tooltip_state
    layer = pdk.Layer(
        "GeoJsonLayer", geo,
        opacity=0.85, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        line_width_min_pixels=1,
        pickable=True, auto_highlight=True,
    )
    return pdk.Deck(layers=[layer], initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v10", tooltip=tip)


st.pydeck_chart(render_map(), use_container_width=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# HOURLY BANDWIDTH PROFILES
# ─────────────────────────────────────────────────────────────
st.subheader("Hourly Upload Bandwidth Profile", anchor=False)
st.caption(
    "Bandwidth is aggregated over NBN-connected EVs only. "
    "Single representative 24-hour day. State and postcode charts are in the state's local standard "
    "time; the national chart is in AEST (UTC+10). Daylight saving is excluded."
)

tab_national, tab_state, tab_postcode = st.tabs(["National", "By State", "By Postcode"])

with tab_national:
    summ = nat_summ
    fig = profile_chart(summ, f"National ({selected_year})", TZ_LABEL["ALL"], YMAX["ALL"], nat_unit)
    st.plotly_chart(fig, use_container_width=True)
    scenario_caption(summ)

with tab_state:
    avail_states = sorted(BY_STATE["state"].unique())
    chosen_state = st.selectbox("Choose state", avail_states, key="state_hourly")
    sub = BY_STATE[(BY_STATE["state"] == chosen_state) & (BY_STATE["year"] == selected_year)]
    summ = summarise_scenarios(sub)
    fig = profile_chart(summ, f"{chosen_state} ({selected_year})", TZ_LABEL[chosen_state],
                        YMAX[chosen_state], pick_unit(YMAX[chosen_state]))
    st.plotly_chart(fig, use_container_width=True)
    scenario_caption(summ)

with tab_postcode:
    suburbs = load_suburbs().set_index("postcode")["SUBURBS_JOINED"]
    avail_pcs = sorted(PC_MEANS.loc[PC_MEANS["year"] == selected_year, "postcode"].unique())
    chosen_pc = st.selectbox(
        "Choose postcode", avail_pcs, key="pc_hourly",
        format_func=lambda p: f"{p} — {suburbs.get(p, '(Unknown)')}",
    )
    reps = load_postcode_reps(selected_year, chosen_pc)
    pc_state = reps["state"].iloc[0] if len(reps) else None
    if pc_state is None:
        st.info("No data for this postcode.")
    else:
        summ = summarise_scenarios(reps)   # one row per replication
        top = max(s["hi_profile"].max() for s in summ.values()) * YLIM_HEADROOM
        fig = profile_chart(summ, f"{chosen_pc} ({suburbs.get(chosen_pc, '(Unknown)')}, {selected_year})",
                            TZ_LABEL.get(pc_state, ""), top, pick_unit(top))
        st.plotly_chart(fig, use_container_width=True)
        scenario_caption(summ)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────
st.subheader("Export", anchor=False)
st.caption("`total_upstream_gb` is from NBN-connected EVs only; `ev_count` is the total fleet.")
st.download_button(
    "Download postcode-level means as CSV",
    data=df_year.to_csv(index=False),
    file_name=f"ev_bandwidth_{selected_scenario}_{selected_year}.csv",
    mime="text/csv",
)