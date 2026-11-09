import ast
import copy

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deterministic Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PHYSICAL / UNIT CONSTANTS  (not user-editable)
# ─────────────────────────────────────────────────────────────
BYTES_TO_GB            = 1000 ** -3        # decimal GB
BYTES_TO_MEGABITS      = 8 / 1_000_000     # decimal, matches real-world Mbps speed tiers
MB_TO_BYTES            = 1000 ** 2
HOURS_PER_SIMULATION   = 24
SECONDS_PER_HOUR       = 3600
SECONDS_PER_SIMULATION = HOURS_PER_SIMULATION * SECONDS_PER_HOUR

BAR_COLOR    = ['#2E86AB', '#A23B72', '#F18F01']
TIER_PALETTE = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#91cf60",
                "#66c2a5", "#3288bd", "#5e4fa2", "#9e0142"]

TELEMETRY_GROUPS = [
    "charging", "driving", "powertrain", "location", "safety",
    "vehicle_state", "media", "climate", "service",
    "vehicle_config", "user_preference",
]
CHARGING_ONLY = {"charging"}
ALWAYS_ACTIVE = {"service", "vehicle_config", "user_preference"}

# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG — one person, single point-values throughout
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "person": {
        "hours_charging":         8.0,
        "hours_driving":          3.0,
        "arrival_time":           18.0,  # Time the vehicle returns home (decimal time)
        "safety_event":           1,     # 0 = no, 1 = yes
        "data_sharing":           1,     # 0 = no, 1 = yes (opted in)
        "base_event_rate":        5.0,   # events / hour
        "adas_usage":             0.5,   # fraction of driving using ADAS
        "clip_payload":           105.0, # event clip size (MB)
        "adas_multiplier":        2.0,
    },
    "telemetry_groups": {
        "charging":        (290.0, 2.000),
        "driving":         (70.0,  2.000),
        "powertrain":      (260.0, 2.000),
        "location":        (100.0,  1.000),
        "safety":          (20.0,  0.500),
        "vehicle_state":   (150.0, 0.500),
        "media":           (140.0,  0.100),
        "climate":         (110.0,  0.100),
        "service":         (150.0, 0.050),
        "vehicle_config":  (170.0,  0.050),
        "user_preference": (25,  0.005),
    },
    "nbn_tiers": {
        "1 Mbps":   1.0,
        "5 Mbps":   5.0,
        "10 Mbps":  10.0,
        "20 Mbps":  20.0,
        "50 Mbps":  50.0,
        "100 Mbps": 100.0,
    },
    "bandwidth_utilisation": {
        "baseline":      1.00,
        "10% reduction": 0.90,
    },
    "selected_tier":     "20 Mbps",
    "selected_scenario": "baseline",
}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "det_config" not in st.session_state:
    st.session_state["det_config"] = copy.deepcopy(DEFAULT_CONFIG)

for _k, _v in {
    "det_results": None, "det_run": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def cfg():
    return st.session_state["det_config"]


# ─────────────────────────────────────────────────────────────
# DETERMINISTIC CORE — single person, exact arithmetic
# ─────────────────────────────────────────────────────────────

def _to_twelve_hour(h):
    """0–48 float hour -> '3:30 AM' style label, wrapping past midnight."""
    h  = h % 24
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        mm, hh = 0, (hh + 1) % 24
    period = "AM" if hh < 12 else "PM"
    hh12   = hh % 12 or 12
    return f"{hh12}:{mm:02d} {period}"


def calculate_usage(person, telemetry):
    """Exact (no sampling) upstream data usage for one person's point-values."""
    hours_charging = person["hours_charging"]
    hours_driving  = person["hours_driving"]

    sec_charging = hours_charging * SECONDS_PER_HOUR
    sec_driving  = hours_driving  * SECONDS_PER_HOUR

    tel = 0.0
    for group in TELEMETRY_GROUPS:
        size, rate = telemetry[group]
        period = (sec_charging if group in CHARGING_ONLY
                  else SECONDS_PER_SIMULATION if group in ALWAYS_ACTIVE
                  else sec_driving)
        tel += size * rate * period

    clip_bytes = person["clip_payload"] * MB_TO_BYTES
    safe = clip_bytes * person["safety_event"]

    w_hrs = (hours_driving * (1 - person["adas_usage"])
             + hours_driving * person["adas_usage"] * person["adas_multiplier"])
    fleet = person["data_sharing"] * person["base_event_rate"] * clip_bytes * w_hrs

    up_total = tel + safe + fleet

    return {
        "upstream_bytes":        up_total,
        "upstream_gb":           up_total * BYTES_TO_GB,
        "upstream_telemetry_gb": tel      * BYTES_TO_GB,
        "upstream_safety_gb":    safe     * BYTES_TO_GB,
        "upstream_fleet_gb":     fleet    * BYTES_TO_GB,
    }


def run_deterministic(config):
    """Compute usage for the single person, then upload time for every NBN
    tier under every scenario (full reference table), plus the headline
    number for the person's selected tier + scenario."""
    person    = config["person"]
    telemetry = config["telemetry_groups"]
    tiers     = config["nbn_tiers"]
    scenarios = config["bandwidth_utilisation"]
    sel_tier  = config["selected_tier"]
    sel_scen  = config["selected_scenario"]

    if not tiers:
        raise ValueError("At least one NBN tier is required.")
    if not scenarios:
        raise ValueError("At least one bandwidth scenario is required.")
    if sel_tier not in tiers:
        raise ValueError(f"Selected tier '{sel_tier}' is not in the tier list.")
    if sel_scen not in scenarios:
        raise ValueError(f"Selected scenario '{sel_scen}' is not in the scenario list.")

    usage = calculate_usage(person, telemetry)
    up_bits = usage["upstream_bytes"] * BYTES_TO_MEGABITS   # megabits
    arrival = person["arrival_time"]

    # Full reference table: every tier x every scenario, including the
    # resulting upload window (start = arrival time, end = arrival + duration)
    rows = []
    for tier_name, speed in tiers.items():
        for scen_name, util in scenarios.items():
            eff_speed = speed * util
            upload_hours = (up_bits / eff_speed) / SECONDS_PER_HOUR
            rows.append({
                "nbn_tier": tier_name, "upload_speed_mbps": speed,
                "scenario": scen_name, "utilisation": util,
                "effective_speed_mbps": eff_speed,
                "upload_hours": upload_hours,
                "upload_start": arrival,
                "upload_end":   arrival + upload_hours,
            })
    full_table = pd.DataFrame(rows)

    # Headline: the person's actual selected tier + scenario
    sel_speed = tiers[sel_tier]
    sel_util  = scenarios[sel_scen]
    sel_eff   = sel_speed * sel_util
    sel_upload_hours = (up_bits / sel_eff) / SECONDS_PER_HOUR
    sel_upload_start = arrival
    sel_upload_end   = arrival + sel_upload_hours

    return {
        "usage":            usage,
        "full_table":       full_table,
        "arrival_time":      arrival,
        "selected_tier":     sel_tier,
        "selected_scenario": sel_scen,
        "selected_speed":    sel_speed,
        "selected_util":     sel_util,
        "selected_eff_speed":sel_eff,
        "selected_upload_hours": sel_upload_hours,
        "selected_upload_start": sel_upload_start,
        "selected_upload_end":   sel_upload_end,
    }


# ─────────────────────────────────────────────────────────────
# CONFIG EDITOR UI
# ─────────────────────────────────────────────────────────────

def render_person_editor():
    p = cfg()["person"]

    st.markdown("**Driver Behaviour**")

    c1, c2, c3 = st.columns(3)
    with c1:
        p["hours_charging"] = st.number_input(
            "Hours charging", 0.0, 24.0, float(p["hours_charging"]), step=0.5, key="d_p_hrs_chg",
        )
    with c2:
        p["hours_driving"] = st.number_input(
            "Hours driving", 0.0, 24.0, float(p["hours_driving"]), step=0.5, key="d_p_hrs_drv",
        )
    with c3:
        p["arrival_time"] = st.number_input(
            "Arrival time (decimal time)", 0.0, 24.0, float(p["arrival_time"]), step=0.5,
            key="d_p_arrival", format="%.2f",
        )

    c4, c5 = st.columns(2)
    with c4:
        p["data_sharing"] = st.selectbox(
            "Data-sharing opt-in", options=[0.0, 1.0],
            index=[0.0, 1.0].index(float(p["data_sharing"])),
            format_func=lambda v: "Yes" if v == 1.0 else "No",
            key="d_p_optin",
        )
    with c5:
        p["safety_event"] = st.selectbox(
            "Safety event", options=[0.0, 1.0], index=[0.0, 1.0].index(float(p["safety_event"])),
            format_func=lambda v: "Yes" if v == 1.0 else "No", key="d_p_safety_event",
        )

    st.markdown("**ADAS Features**")
    c6, c7, c8 = st.columns(3)
    with c6:
        p["adas_multiplier"] = st.slider(
            "ADAS multiplier", 1.0, 5.0, float(p["adas_multiplier"]), step=0.5, key="d_p_adas_mult",
        )
    with c7:
        p["base_event_rate"] = st.slider(
            "Base events / hour", 0.0, 15.0, float(p["base_event_rate"]), step=0.5, key="d_p_base_ev",
        )
    with c8:
        p["adas_usage"] = st.slider(
            "ADAS usage fraction", 0.0, 1.0, float(p["adas_usage"]), step=0.05, key="d_p_adas",
        )

    st.markdown("**Clip Configuration**")

    p["clip_payload"] = st.slider(
        "Clip payload (MB)", 60.0, 150.0, float(p["clip_payload"]), step=5.0, key="d_p_clip_payload",
    )


def render_telemetry_editor():
    st.caption("For each telemetry group: message size in bytes, message rate in Hz.")
    tg = cfg()["telemetry_groups"]
    for group in TELEMETRY_GROUPS:
        size, rate = tg[group]
        st.markdown(f"**{group.replace('_', ' ').title()}**")
        tc1, tc2 = st.columns(2)
        with tc1:
            new_size = st.slider(
                "Size (bytes)", 0.0, 300.0, float(size), step=1.0, key=f"d_tg_size_{group}",
            )
        with tc2:
            new_rate = st.slider(
                "Rate (Hz)", 0.0, 2.000, float(rate), step=0.005, key=f"d_tg_rate_{group}", format="%.4f",
            )
        tg[group] = (new_size, new_rate)


def render_tier_and_scenario_editor():
    tiers     = cfg()["nbn_tiers"]
    scenarios = cfg()["bandwidth_utilisation"]

    st.markdown("##### NBN Tier")
    st.caption("Choose the person's upload speed.")

    tier_names = list(tiers.keys())
    sel_idx = tier_names.index(cfg()["selected_tier"]) if cfg()["selected_tier"] in tier_names else 0
    cfg()["selected_tier"] = st.selectbox(
        "Selected NBN tier", options=tier_names, index=sel_idx, key="d_selected_tier",
    )

    st.markdown("---")
    st.markdown("##### Bandwidth Utilisation Scenario")
    st.caption(
        "Choose the scenario applied to the person's tier. "
        "A scenario scales the tier's effective upload speed by a factor "
        "(e.g. 0.90 = 10% reduction from congestion)."
    )

    scen_names = list(scenarios.keys())
    sel_idx2 = scen_names.index(cfg()["selected_scenario"]) if cfg()["selected_scenario"] in scen_names else 0
    cfg()["selected_scenario"] = st.selectbox(
        "Selected scenario", options=scen_names, index=sel_idx2, key="d_selected_scenario",
    )


# ─────────────────────────────────────────────────────────────
# MANUAL DICTIONARY INPUT
# ─────────────────────────────────────────────────────────────
_DICT_REQUIRED_TOP_KEYS = ["PERSON", "TELEMETRY_GROUPS", "NBN_TIERS", "BANDWIDTH_UTILISATION",
                           "SELECTED_TIER", "SELECTED_SCENARIO"]

_DEFAULT_DICT_STR = """{
    'PERSON': {
        'hours_charging':         8,
        'hours_driving':          3,
        'arrival_time':           18,    # 24h clock, hour the vehicle returns home
        'safety_event':               1,
        'data_sharing':           1,    # 0 = no, 1 = yes (opted in)
        'base_event_rate':        5,
        'adas_usage':             0.5,
        'clip_payload':           105,
        'adas_multiplier':        2,
    },

    'TELEMETRY_GROUPS': {
        # (size_B, rate_Hz)
        'charging':        (241.0, 1.5),
        'driving':         (54.0,  1.5),
        'powertrain':      (200.0, 1.5),
        'location':        (70.0,  0.75),
        'safety':          (18.0,  0.3),
        'vehicle_state':   (121.0, 0.3),
        'media':           (80.0,  0.075),
        'climate':         (97.0,  0.075),
        'service':         (118.0, 0.0275),
        'vehicle_config':  (90.0,  0.0275),
        'user_preference': (12.5,  0.00275),
    },

    'NBN_TIERS': {
        '1 Mbps':   1,
        '5 Mbps':   5,
        '10 Mbps':  10,
        '20 Mbps':  20,
        '50 Mbps':  50,
        '100 Mbps': 100,
    },

    'BANDWIDTH_UTILISATION': {
        'baseline':      1.00,
        '10% reduction': 0.90,
    },

    'SELECTED_TIER':     '20 Mbps',
    'SELECTED_SCENARIO': 'baseline',
}"""


def _parse_dict_to_config(raw_text):
    try:
        parsed = ast.literal_eval(raw_text.strip())
    except Exception as e:
        raise ValueError(f"Could not parse as a Python dict literal: {e}")

    if not isinstance(parsed, dict):
        raise ValueError("Input must be a single Python dict literal.")

    missing = [k for k in _DICT_REQUIRED_TOP_KEYS if k not in parsed]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")

    # ── PERSON ───────────────────────────────────────────────
    raw_person = parsed["PERSON"]
    if not isinstance(raw_person, dict):
        raise ValueError("PERSON must be a dict.")

    _required_person_keys = [
        "hours_charging", "hours_driving", "arrival_time", "safety_event",
        "data_sharing", "base_event_rate", "adas_usage",
        "clip_payload", "adas_multiplier",
    ]
    person = {}
    for key in _required_person_keys:
        if key not in raw_person:
            raise ValueError(f"PERSON is missing '{key}'.")
        v = raw_person[key]
        if isinstance(v, tuple):
            raise ValueError(f"PERSON['{key}'] must be a single number, got {v!r}.")
        person[key] = int(v) if key in ("safety_event", "data_sharing") else float(v)

    # ── TELEMETRY_GROUPS ─────────────────────────────────────
    raw_tg = parsed["TELEMETRY_GROUPS"]
    if not isinstance(raw_tg, dict):
        raise ValueError("TELEMETRY_GROUPS must be a dict.")
    telemetry = {}
    for group in TELEMETRY_GROUPS:
        if group not in raw_tg:
            raise ValueError(f"TELEMETRY_GROUPS is missing '{group}'.")
        v = raw_tg[group]
        if not (isinstance(v, tuple) and len(v) == 2):
            raise ValueError(f"TELEMETRY_GROUPS['{group}'] must be a 2-tuple (size, rate), got {v!r}.")
        telemetry[group] = (float(v[0]), float(v[1]))

    # ── NBN_TIERS ────────────────────────────────────────────
    raw_tiers = parsed["NBN_TIERS"]
    if not isinstance(raw_tiers, dict) or not raw_tiers:
        raise ValueError("NBN_TIERS must be a non-empty dict.")
    tiers = {}
    for name, speed in raw_tiers.items():
        if isinstance(speed, tuple):
            raise ValueError(f"NBN_TIERS['{name}'] must be a single number (Mbps), got {speed!r}.")
        tiers[name] = float(speed)

    # ── BANDWIDTH_UTILISATION ─────────────────────────────────
    raw_bw = parsed["BANDWIDTH_UTILISATION"]
    if not isinstance(raw_bw, dict) or not raw_bw:
        raise ValueError("BANDWIDTH_UTILISATION must be a non-empty dict.")
    bandwidth = {str(k): float(v) for k, v in raw_bw.items()}

    # ── SELECTED_TIER / SELECTED_SCENARIO ─────────────────────
    sel_tier = parsed["SELECTED_TIER"]
    sel_scen = parsed["SELECTED_SCENARIO"]
    if sel_tier not in tiers:
        raise ValueError(f"SELECTED_TIER '{sel_tier}' is not a key in NBN_TIERS.")
    if sel_scen not in bandwidth:
        raise ValueError(f"SELECTED_SCENARIO '{sel_scen}' is not a key in BANDWIDTH_UTILISATION.")

    return {
        "person":                person,
        "telemetry_groups":      telemetry,
        "nbn_tiers":              tiers,
        "bandwidth_utilisation":  bandwidth,
        "selected_tier":          sel_tier,
        "selected_scenario":      sel_scen,
    }


def _fmt_num(x):
    if isinstance(x, float) and x == int(x) and abs(x) < 1e6:
        return f"{x:.1f}"
    return repr(round(x, 6) if isinstance(x, float) else x)


def _config_to_dict_str(config):
    lines = ["{"]
    p = config["person"]
    lines.append("    'PERSON': {")
    lines.append(f"        'hours_charging':         {_fmt_num(p['hours_charging'])},")
    lines.append(f"        'hours_driving':          {_fmt_num(p['hours_driving'])},")
    lines.append(f"        'arrival_time':           {_fmt_num(p['arrival_time'])},    # 24h clock, hour the vehicle returns home")
    lines.append(f"        'safety_event':               {int(p['safety_event'])},")
    lines.append(f"        'data_sharing':           {int(p['data_sharing'])},    # 0 = no, 1 = yes (opted in)")
    lines.append(f"        'base_event_rate':        {_fmt_num(p['base_event_rate'])},")
    lines.append(f"        'adas_usage':             {_fmt_num(p['adas_usage'])},")
    lines.append(f"        'clip_payload':           {_fmt_num(p['clip_payload'])},")
    lines.append(f"        'adas_multiplier':        {_fmt_num(p['adas_multiplier'])},")
    lines.append("    },")
    lines.append("")

    tg = config["telemetry_groups"]
    lines.append("    'TELEMETRY_GROUPS': {")
    lines.append("        # (size_B, rate_Hz)")
    for group in TELEMETRY_GROUPS:
        pad = " " * max(0, 17 - len(group))
        size, rate = tg[group]
        lines.append(f"        '{group}':{pad}({_fmt_num(size)}, {_fmt_num(rate)}),")
    lines.append("    },")
    lines.append("")

    lines.append("    'NBN_TIERS': {")
    tier_items = list(config["nbn_tiers"].items())
    for i, (name, speed) in enumerate(tier_items):
        comma = "," if i < len(tier_items) - 1 else ""
        lines.append(f"        '{name}': {_fmt_num(speed)}{comma}")
    lines.append("    },")
    lines.append("")

    lines.append("    'BANDWIDTH_UTILISATION': {")
    bw_items = list(config["bandwidth_utilisation"].items())
    for i, (name, util) in enumerate(bw_items):
        comma = "," if i < len(bw_items) - 1 else ""
        lines.append(f"        '{name}': {_fmt_num(util)}{comma}")
    lines.append("    },")
    lines.append("")
    lines.append(f"    'SELECTED_TIER':     '{config['selected_tier']}',")
    lines.append(f"    'SELECTED_SCENARIO': '{config['selected_scenario']}',")
    lines.append("}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────

def render_results(result):
    usage = result["usage"]
    arrival = result["arrival_time"]

    st.caption(
        f"Arrival time: **{_to_twelve_hour(arrival)}**  ·  "
        f"Upload Speed: **{result['selected_tier']}** ({result['selected_speed']:.0f} Mbps)  ·  "
        f"Scenario: **{result['selected_scenario']}** (×{result['selected_util']:.2f})  ·  "
        f"Effective speed: **{result['selected_eff_speed']:.2f} Mbps**"
    )
    st.markdown("---")

    # ── Headline KPIs ─────────────────────────────────────────
    st.subheader("Upstream Data and Upload Time", anchor=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Total upstream (GB/day)", f"{usage['upstream_gb']:.4f}")
    k2.metric("Upload time (hours)",     f"{result['selected_upload_hours']:.4f}")
    k3.metric("Upload completes at",     _to_twelve_hour(result["selected_upload_end"]))

    st.markdown("---")

    # ── Breakdown ─────────────────────────────────────────────
    st.subheader("Upstream Data Breakdown", anchor=False)
    components = {
        "Telemetry":     usage["upstream_telemetry_gb"],
        "Safety clips":  usage["upstream_safety_gb"],
        "Fleet sharing": usage["upstream_fleet_gb"],
    }
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=list(components.keys()), y=list(components.values()),
        marker_color=[BAR_COLOR[0], BAR_COLOR[1], BAR_COLOR[2]],
        marker_line_color="black", marker_line_width=0.6, opacity=0.7,
        text=[f"{v:.4f} GB" for v in components.values()], textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis=dict(title="Component"),
        yaxis=dict(title="Upstream GB / day", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Full tier x scenario reference table ─────────────────
    st.subheader("Upload Time Across All Tiers and Scenarios", anchor=False)

    full = result["full_table"].copy()
    pivot = full.pivot(index="nbn_tier", columns="scenario", values="upload_hours")
    tier_order = list(cfg()["nbn_tiers"].keys())
    pivot = pivot.reindex([t for t in tier_order if t in pivot.index])

    fig_tier = go.Figure()

    for i, scen in enumerate(pivot.columns):
        fig_tier.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[scen],
            name=scen,
            marker_color=[
                TIER_PALETTE[j % len(TIER_PALETTE)]
                for j in range(len(pivot.index))
            ],
            opacity=1.0 if i == 0 else 0.6,
            marker_line_color="black",
            marker_line_width=0.4,
            text=scen
        ))
    fig_tier.update_layout(
        barmode="group",
        xaxis=dict(title="Upload Speed"),
        yaxis=dict(title="Upload time (hours)", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")

    # ── Export ────────────────────────────────────────────────
    st.subheader("Export", anchor=False)
    st.download_button(
        "Download full tier and scenario table as CSV",
        data=full.to_csv(index=False),
        file_name="deterministic_single_person_tiers.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────
st.title("Deterministic Model", anchor=False)
st.markdown(
    "Single-person, single-point version of the EV upstream data model."
)

with st.expander("Model Input", expanded=True):
    sub_tabs = st.tabs(["User Parameters", "Upload Speed and Scenario", "Telemetry Groups"])
    with sub_tabs[0]:
        render_person_editor()
    with sub_tabs[1]:
        render_tier_and_scenario_editor()
    with sub_tabs[2]:
        render_telemetry_editor()

    st.markdown("---")
    run_clicked = st.button("▶  Calculate", type="primary", use_container_width=True, key="d_run_btn")


if run_clicked:
    errors = []
    if not cfg()["nbn_tiers"]:
        errors.append("At least one NBN tier is required.")
    if not cfg()["bandwidth_utilisation"]:
        errors.append("At least one bandwidth scenario is required.")
    if cfg()["nbn_tiers"] and cfg()["selected_tier"] not in cfg()["nbn_tiers"]:
        errors.append("Selected NBN tier is no longer valid — please re-select.")
    if cfg()["bandwidth_utilisation"] and cfg()["selected_scenario"] not in cfg()["bandwidth_utilisation"]:
        errors.append("Selected scenario is no longer valid — please re-select.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        config_snapshot = copy.deepcopy(cfg())
        result = run_deterministic(config_snapshot)
        st.session_state["det_results"] = result
        st.session_state["det_run"]     = True

if st.session_state["det_run"] and st.session_state["det_results"] is not None:
    render_results(st.session_state["det_results"])