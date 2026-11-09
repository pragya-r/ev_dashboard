import ast
import copy
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import truncnorm

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Simulation",
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
PROFILE_HOURS          = 48                # busy-hour window (covers next-day spillover)

BAR_COLOR    = "#A23B72"
TIER_PALETTE = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#91cf60",
                "#66c2a5", "#3288bd", "#5e4fa2", "#9e0142"]  # extends if >6 tiers

TELEMETRY_GROUPS = [
    "charging", "driving", "powertrain", "location", "safety",
    "vehicle_state", "media", "climate", "service",
    "vehicle_config", "user_preference",
]
CHARGING_ONLY = {"charging"}
ALWAYS_ACTIVE = {"service", "vehicle_config", "user_preference"}

MIXTURE_KEYS   = ("w", "a1", "b1", "a2", "b2", "scale")
MIXTURE_LABELS = {
    "home_arrival_time": "Home arrival time (hours, 0–24)",
    "hours_driving":     "Driving time (hours)",
    "hours_charging":    "Charging time (hours)",
}

# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG  (mirrors traffic_model.ipynb)
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "behaviour": {
        "safety_event_p":      1.0,           # Binomial(1, p) per EV per day
        "data_sharing_p":      1.0,           # Binomial(1, p) per EV
        "driving_probability": 1.0,           # P(EV drives that day); 1.0 = every day
        "tier_performance":    (0.6, 0.7),    # Uniform(lo, hi) share of tier speed achieved
    },
    # Two-component Beta mixtures:  scale * [ w·Beta(a1,b1) + (1-w)·Beta(a2,b2) ]
    "mixtures": {
        "home_arrival_time": {"w": 0.90, "a1": 7.5,  "b1": 2.5,   "a2": 0.5,  "b2": 0.5,   "scale": 24.0},
        "hours_driving":     {"w": 0.03, "a1": 0.94, "b1": 2.64,  "a2": 1.38, "b2": 15.05, "scale": 12.5},
        "hours_charging":    {"w": 0.03, "a1": 8.34, "b1": 1.81,  "a2": 1.69, "b2": 4.81,  "scale": 8.5},
    },
    "shared_ranges": {
        "base_event_rate": (2.0,   15.0),   # events / hour
        "adas_usage":      (0.0,    1.0),   # fraction of driving using ADAS
        "clip_payload":    (60.0, 150.0),   # event clip size (MB)
        "adas_multiplier": (2.0,    5.0),
    },
    "telemetry_groups": {
        "charging":        (191.0, 291.0, 1.0,    2.0),
        "driving":         (38.0,   70.0, 1.0,    2.0),
        "powertrain":      (140.0, 260.0, 1.0,    2.0),
        "location":        (40.0,  100.0, 0.5,    1.0),
        "safety":          (12.0,   24.0, 0.1,    0.5),
        "vehicle_state":   (100.0, 142.0, 0.1,    0.5),
        "media":           (24.0,  136.0, 0.05,   0.1),
        "climate":         (89.0,  105.0, 0.05,   0.1),
        "service":         (92.0,  144.0, 0.005,  0.05),
        "vehicle_config":  (8.0,   172.0, 0.005,  0.05),
        "user_preference": (0.0,    25.0, 0.0005, 0.005),
    },
    "nbn_tiers": {
        "1 Mbps":   {"upload_speed_mbps": 1.0,   "weight": 0.05},
        "5 Mbps":   {"upload_speed_mbps": 5.0,   "weight": 0.14},
        "10 Mbps":  {"upload_speed_mbps": 10.0,  "weight": 0.07},
        "20 Mbps":  {"upload_speed_mbps": 20.0,  "weight": 0.40},
        "50 Mbps":  {"upload_speed_mbps": 50.0,  "weight": 0.28},
        "100 Mbps": {"upload_speed_mbps": 100.0, "weight": 0.06},
    },
    # Scenario factor applied to tier speed. Whole-transfer in the per-tier upload
    # times; peak hours only in the busy-hour profile.
    "bandwidth_utilisation": {
        "baseline":      1.00,
        "10% reduction": 0.90,
        "15% reduction": 0.85,
        "20% reduction": 0.80,
        "25% reduction": 0.75,
    },
    "peak_hours": (19, 23),   # local clock hours [start, end) — notebook: range(19, 23)
}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state["config"] = copy.deepcopy(DEFAULT_CONFIG)

for _k, _v in {
    "mc_results": None, "mc_profile": None, "mc_elapsed": None,
    "mc_scenario": None, "mc_n": None, "mc_run": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def cfg():
    return st.session_state["config"]


# ─────────────────────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────────────────────

def _norm_dist(rng, low, high, size):
    """Truncated normal on [low, high]; mu = midpoint, sigma = range / 4."""
    if low == high:
        return np.full(size, float(low))
    mu, sigma = (low + high) / 2, (high - low) / 4
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)


def _sample_beta_mixture(rng, n, w, a1, b1, a2, b2, scale):
    comp = rng.binomial(1, w, size=n).astype(bool)
    samples = np.empty(n)
    samples[ comp] = rng.beta(a1, b1, size=comp.sum())
    samples[~comp] = rng.beta(a2, b2, size=(~comp).sum())
    return scale * samples


def _weighted_choice(rng, keys, weights, n):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(keys, size=n, p=weights)


def _bytes_to_hours(upstream_bytes, speed_mbps):
    """Byte count -> upload time (hours) at `speed_mbps`."""
    return (upstream_bytes * BYTES_TO_MEGABITS / speed_mbps) / SECONDS_PER_HOUR


def generate_inputs(rng, n, config):
    beh      = config["behaviour"]
    mix      = config["mixtures"]
    shared   = config["shared_ranges"]
    tiers_cfg = config["nbn_tiers"]

    if not tiers_cfg:
        raise ValueError("At least one NBN tier is required.")

    tier_names = list(tiers_cfg.keys())
    tier_w     = [tiers_cfg[t]["weight"] for t in tier_names]
    tiers      = _weighted_choice(rng, tier_names, tier_w, n)
    up_speed   = np.array([tiers_cfg[t]["upload_speed_mbps"] for t in tiers], dtype=float)

    hours_driving  = _sample_beta_mixture(rng, n, **mix["hours_driving"])
    hours_charging = _sample_beta_mixture(rng, n, **mix["hours_charging"])
    safety_event   = rng.binomial(1, beh["safety_event_p"], size=n)

    if beh["driving_probability"] < 1.0:
        is_driving_day = rng.binomial(1, beh["driving_probability"], size=n)
        drove = is_driving_day.astype(bool)
        hours_driving[~drove] = 0.0
        safety_event[~drove]  = 0
    else:
        is_driving_day = np.ones(n)

    # NOTE: draw order below matches traffic_model.ipynb so a given seed reproduces it.
    s = {
        "is_driving_day":    is_driving_day,
        "speed_tier":        tiers,
        "upload_speed_mbps": up_speed,
        "tier_performance":  rng.uniform(*beh["tier_performance"], size=n),
        "home_arrival_time": _sample_beta_mixture(rng, n, **mix["home_arrival_time"]),
        "hours_driving":     hours_driving,
        "hours_charging":    hours_charging,
        "safety_event":      safety_event,
        "base_event_rate":   _norm_dist(rng, *shared["base_event_rate"], n),
        "data_sharing":      rng.binomial(1, beh["data_sharing_p"], n),
        "adas_usage":        _norm_dist(rng, *shared["adas_usage"], n),
        "clip_payload":      _norm_dist(rng, *shared["clip_payload"], n),
        "adas_multiplier":   _norm_dist(rng, *shared["adas_multiplier"], n),
    }

    for group in TELEMETRY_GROUPS:
        sz_lo, sz_hi, rt_lo, rt_hi = config["telemetry_groups"][group]
        s[f"{group}_size"] = _norm_dist(rng, sz_lo, sz_hi, n)
        s[f"{group}_rate"] = _norm_dist(rng, rt_lo, rt_hi, n)

    return pd.DataFrame(s)


def calc_upstream_telemetry(df):
    sec_charging = df["hours_charging"].values * SECONDS_PER_HOUR
    sec_driving  = df["hours_driving"].values  * SECONDS_PER_HOUR
    total = np.zeros(len(df))
    for group in TELEMETRY_GROUPS:
        sz = df[f"{group}_size"].values
        rt = df[f"{group}_rate"].values
        period = (sec_charging if group in CHARGING_ONLY
                  else SECONDS_PER_SIMULATION if group in ALWAYS_ACTIVE
                  else sec_driving)
        total += sz * rt * period
    return total


def calc_upstream_safety(df):
    clip_bytes = df["clip_payload"].values * MB_TO_BYTES
    return clip_bytes * df["safety_event"].values


def calc_upstream_fleet(df):
    clip_bytes = df["clip_payload"].values * MB_TO_BYTES
    h_driving  = df["hours_driving"].values
    adas_frac  = df["adas_usage"].values
    weighted_hrs = (h_driving * (1 - adas_frac)
                    + h_driving * adas_frac * df["adas_multiplier"].values)
    return df["data_sharing"].values * df["base_event_rate"].values * clip_bytes * weighted_hrs


def calculate_data_usage(df):
    tel  = calc_upstream_telemetry(df)
    safe = calc_upstream_safety(df)
    flt  = calc_upstream_fleet(df)
    up_tot = tel + safe + flt
    return pd.DataFrame({
        "upstream_bytes":        up_tot,
        "upstream_gb":           up_tot * BYTES_TO_GB,
        "upstream_telemetry_gb": tel    * BYTES_TO_GB,
        "upstream_safety_gb":    safe   * BYTES_TO_GB,
        "upstream_fleet_gb":     flt    * BYTES_TO_GB,
    })


def analyse_by_tier(usage_df, scenario, config):
    util = config["bandwidth_utilisation"].get(scenario, 1.0)
    return pd.DataFrame({
        f"{tier}_upload_hours": _bytes_to_hours(usage_df["upstream_bytes"].values,
                                                info["upload_speed_mbps"] * util)
        for tier, info in config["nbn_tiers"].items()
    })


@st.cache_data(show_spinner=False)
def run_simulation(n, scenario, seed, config, _cache_key):
    rng = np.random.default_rng(seed)
    inputs = generate_inputs(rng, n, config)
    usage  = calculate_data_usage(inputs)
    tiers  = analyse_by_tier(usage, scenario, config)
    return pd.concat([inputs, usage, tiers], axis=1)


def _config_cache_key(config, n, scenario, seed):
    """Hashable repr so st.cache_data invalidates whenever config changes."""
    return repr((n, scenario, seed, sorted(config.items())))


# ─────────────────────────────────────────────────────────────
# BUSY-HOUR PROFILE  (slot-by-slot over PROFILE_HOURS)
# ─────────────────────────────────────────────────────────────

def compute_hourly_profile(results, scenario, config):
    """Per-hour active-EV count and aggregate bandwidth (Mbps).

    Each EV uploads continuously from home_arrival_time at
    upload_speed_mbps * tier_performance until its payload is sent. During
    peak hours the scenario factor scales that speed. Returns (profile, finish)
    where finish is each EV's upload completion time (hours since midnight of
    the arrival day; PROFILE_HOURS if not finished within the window).
    """
    speed     = results["upload_speed_mbps"].to_numpy() * results["tier_performance"].to_numpy()
    start     = results["home_arrival_time"].to_numpy()
    remaining = results["upstream_bytes"].to_numpy() * BYTES_TO_MEGABITS   # megabits
    busy      = config["bandwidth_utilisation"].get(scenario, 1.0)
    p0, p1    = config["peak_hours"]
    peak      = set(range(int(p0), int(p1)))

    finish    = np.where(remaining > 1e-9, np.nan, start)
    ev_count  = np.zeros(PROFILE_HOURS, dtype=np.int64)
    bandwidth = np.zeros(PROFILE_HOURS, dtype=np.float64)

    for h in range(PROFILE_HOURS):
        slot_speed = speed * (busy if (h % 24) in peak else 1.0)

        started = start < (h + 1)
        active  = started & (remaining > 1e-9)
        ev_count[h] = active.sum()

        if not np.any(remaining > 1e-9):
            break
        if not np.any(active):
            continue

        eff_start    = np.maximum(start, h)
        time_in_slot = np.clip((h + 1) - eff_start, 0.0, 1.0)
        time_in_slot = np.where(active, time_in_slot, 0.0)

        capacity = slot_speed * time_in_slot * SECONDS_PER_HOUR
        consumed = np.where(active, np.minimum(capacity, remaining), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            used_hours = np.where(slot_speed > 0, consumed / (slot_speed * SECONDS_PER_HOUR), 0.0)
        used_hours = np.nan_to_num(used_hours)

        bandwidth[h] += (slot_speed * used_hours)[active].sum()
        remaining = remaining - consumed

        done = active & (remaining <= 1e-9)
        finish[done] = (eff_start + used_hours)[done]

    finish = np.where(np.isnan(finish), float(PROFILE_HOURS), finish)
    profile = pd.DataFrame({"hour": np.arange(PROFILE_HOURS),
                            "ev_count": ev_count, "bandwidth_mbps": bandwidth})
    return profile, finish


# ─────────────────────────────────────────────────────────────
# HELPERS (stats / formatting)
# ─────────────────────────────────────────────────────────────

def _stats(series):
    return {k: series.quantile(q) if k not in ("mean",) else series.mean()
            for k, q in [("mean", None), ("median", 0.5),
                         ("p5", 0.05), ("p25", 0.25), ("p75", 0.75), ("p95", 0.95)]}


def _to_twelve_hour(h):
    h  = h % 24
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        mm, hh = 0, (hh + 1) % 24
    period = "AM" if hh < 12 else "PM"
    hh12   = hh % 12 or 12
    return f"{hh12}:{mm:02d} {period}"


# ─────────────────────────────────────────────────────────────
# CONFIG EDITOR UI
# ─────────────────────────────────────────────────────────────

def render_behaviour_editor():
    st.caption(
        "Per-EV behaviour. Driving and charging hours and home-arrival time are drawn from "
        "Beta mixtures (see Advanced Parameters); the probabilities below are Binomial(1, p) draws."
    )
    b = cfg()["behaviour"]

    c1, c2, c3 = st.columns(3)
    with c1:
        b["safety_event_p"] = st.number_input(
            "Safety event (probability)", min_value=0.0, max_value=1.0,
            value=float(b["safety_event_p"]), step=0.05, key="beh_safety_p",
        )
    with c2:
        b["data_sharing_p"] = st.number_input(
            "Data-sharing opt-in (probability)", min_value=0.0, max_value=1.0,
            value=float(b["data_sharing_p"]), step=0.05, key="beh_share_p",
        )
    with c3:
        b["driving_probability"] = st.number_input(
            "Driving-day probability", min_value=0.0, max_value=1.0,
            value=float(b["driving_probability"]), step=0.05, key="beh_drive_p",
        )
    st.caption(
        "Driving-day probability < 1 zeroes driving hours and safety events on non-driving days. "
        "1.0 = every EV drives every day."
    )

    b["tier_performance"] = tuple(st.slider(
        "Tier performance (fraction of tier speed achieved)", 0.0, 1.0,
        tuple(float(x) for x in b["tier_performance"]), step=0.01, key="beh_tier_perf",
    ))


def render_advanced_editor():
    st.caption(
        "These parameters apply to every EV. Adjust only if you understand the underlying model."
    )
    with st.expander("Behavioural ranges", expanded=False):
        sr = cfg()["shared_ranges"]

        c1, c2 = st.columns(2)
        with c1:
            sr["base_event_rate"] = tuple(st.slider(
                "Base events / hour", 0.0, 30.0, tuple(sr["base_event_rate"]),
                step=0.5, key="sr_base_ev",
            ))
            sr["clip_payload"] = tuple(st.slider(
                "Clip payload (MB)", 10.0, 300.0, tuple(sr["clip_payload"]),
                step=5.0, key="sr_clip_payload",
            ))
        with c2:
            sr["adas_usage"] = tuple(st.slider(
                "ADAS usage fraction", 0.0, 1.0, tuple(sr["adas_usage"]),
                step=0.05, key="sr_adas",
            ))
            sr["adas_multiplier"] = tuple(st.slider(
                "ADAS multiplier", 1.0, 10.0, tuple(sr["adas_multiplier"]),
                step=0.5, key="sr_adas_mult",
            ))

    with st.expander("Time distributions (Beta mixtures)", expanded=False):
        st.caption(
            "value = scale × [ w · Beta(a1, b1) + (1 − w) · Beta(a2, b2) ]. "
            "`w` is the probability of drawing from the first component."
        )
        mix = cfg()["mixtures"]
        for name, label in MIXTURE_LABELS.items():
            m = mix[name]
            st.markdown(f"**{label}**")
            cols = st.columns(6)
            m["w"]     = cols[0].number_input("w", 0.0, 1.0, float(m["w"]), step=0.01, key=f"mix_{name}_w")
            m["a1"]    = cols[1].number_input("a1", 0.01, 100.0, float(m["a1"]), step=0.1, key=f"mix_{name}_a1")
            m["b1"]    = cols[2].number_input("b1", 0.01, 100.0, float(m["b1"]), step=0.1, key=f"mix_{name}_b1")
            m["a2"]    = cols[3].number_input("a2", 0.01, 100.0, float(m["a2"]), step=0.1, key=f"mix_{name}_a2")
            m["b2"]    = cols[4].number_input("b2", 0.01, 100.0, float(m["b2"]), step=0.1, key=f"mix_{name}_b2")
            m["scale"] = cols[5].number_input("scale", 0.1, 100.0, float(m["scale"]), step=0.5, key=f"mix_{name}_scale")

    # (size in bytes / msg, rate in Hz)
    with st.expander("Telemetry group ranges", expanded=False):
        tg = cfg()["telemetry_groups"]
        st.caption(
            "For each telemetry group, message sizes are measured in bytes and message rate in Hz. "
        )
        for group in TELEMETRY_GROUPS:
            sz_lo, sz_hi, rt_lo, rt_hi = tg[group]
            st.markdown(f"**{group.replace('_', ' ').title()}**")
            tc1, tc2 = st.columns(2)
            with tc1:
                new_sz = st.slider(
                    "Size range (bytes)", 0.0, 400.0, (float(sz_lo), float(sz_hi)),
                    step=1.0, key=f"tg_size_{group}",
                )
            with tc2:
                new_rt = st.slider(
                    "Rate range (Hz)", 0.0, 5.0, (float(rt_lo), float(rt_hi)),
                    step=0.005, key=f"tg_rate_{group}", format="%.4f",
                )
            tg[group] = (new_sz[0], new_sz[1], new_rt[0], new_rt[1])


# ─────────────────────────────────────────────────────────────
# MANUAL DICTIONARY INPUT
# ─────────────────────────────────────────────────────────────
_DICT_REQUIRED_TOP_KEYS = ["INPUTS", "MIXTURES", "SPEED_DIST", "CONGESTION", "PEAK_HOURS"]

_DEFAULT_DICT_STR = """{
    'INPUTS': {
        'safety_event':        1,            # Binomial(1, p)
        'driving_probability': 1.0,          # P(EV drives that day); 1.0 = every day
        'data_sharing':        1,            # Binomial(1, p)
        'tier_performance':    (0.6, 0.7),   # Uniform(lo, hi)
        'base_event_rate':     (2, 15),      # per hour
        'adas_usage':          (0, 1),       # fraction of driving using ADAS
        'clip_payload':        (60, 150),    # event clip size (MB)
        'adas_multiplier':     (2, 5),
        # Telemetry groups: (size_min_B, size_max_B, rate_min_Hz, rate_max_Hz)
        'charging':       (191, 291, 1.0,    2.0),
        'driving':        (38,   70, 1.0,    2.0),
        'powertrain':     (140, 260, 1.0,    2.0),
        'location':       (40,  100, 0.5,    1.0),
        'safety':         (12,   24, 0.1,    0.5),
        'vehicle_state':  (100, 142, 0.1,    0.5),
        'media':          (24,  136, 0.05,   0.1),
        'climate':        (89,  105, 0.05,   0.1),
        'service':        (92,  144, 0.005,  0.05),
        'vehicle_config': (8,   172, 0.005,  0.05),
        'user_preference':(0,    25, 0.0005, 0.005),
    },

    # scale * [ w*Beta(a1,b1) + (1-w)*Beta(a2,b2) ]
    'MIXTURES': {
        'home_arrival_time': {'w': 0.90, 'a1': 7.5,  'b1': 2.5,  'a2': 0.5,  'b2': 0.5,   'scale': 24},
        'hours_driving':     {'w': 0.03, 'a1': 0.94, 'b1': 2.64, 'a2': 1.38, 'b2': 15.05, 'scale': 12.5},
        'hours_charging':    {'w': 0.03, 'a1': 8.34, 'b1': 1.81, 'a2': 1.69, 'b2': 4.81,  'scale': 8.5},
    },

    'SPEED_DIST': {
        '1 Mbps':   {'upload_speed_mbps': 1,   'weight': 0.05},
        '5 Mbps':   {'upload_speed_mbps': 5,   'weight': 0.14},
        '10 Mbps':  {'upload_speed_mbps': 10,  'weight': 0.07},
        '20 Mbps':  {'upload_speed_mbps': 20,  'weight': 0.40},
        '50 Mbps':  {'upload_speed_mbps': 50,  'weight': 0.28},
        '100 Mbps': {'upload_speed_mbps': 100, 'weight': 0.06},
    },

    'CONGESTION': {
        'baseline':      1.00,
        '10% reduction': 0.90,
        '15% reduction': 0.85,
        '20% reduction': 0.80,
        '25% reduction': 0.75,
    },

    # Local clock hours [start, end) during which the CONGESTION factor applies
    # in the busy-hour profile (notebook: range(19, 23)).
    'PEAK_HOURS': (19, 23),
}"""


def _two_tuple(v, label):
    if not (isinstance(v, tuple) and len(v) == 2):
        raise ValueError(f"{label} must be a 2-tuple, got {v!r}.")
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        raise ValueError(f"{label}: min > max.")
    return lo, hi


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

    # ── INPUTS ───────────────────────────────────────────────
    raw_in = parsed["INPUTS"]
    if not isinstance(raw_in, dict):
        raise ValueError("INPUTS must be a dict.")

    def _prob(key):
        if key not in raw_in:
            raise ValueError(f"INPUTS is missing '{key}'.")
        p = raw_in[key]
        if isinstance(p, (tuple, list, dict)) or not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"INPUTS['{key}'] must be a probability in [0, 1], got {p!r}.")
        return float(p)

    behaviour = {
        "safety_event_p":      _prob("safety_event"),
        "data_sharing_p":      _prob("data_sharing"),
        "driving_probability": _prob("driving_probability"),
    }
    if "tier_performance" not in raw_in:
        raise ValueError("INPUTS is missing 'tier_performance'.")
    tp = _two_tuple(raw_in["tier_performance"], "INPUTS['tier_performance']")
    if not (0.0 <= tp[0] and tp[1] <= 1.0):
        raise ValueError("INPUTS['tier_performance'] must lie within [0, 1].")
    behaviour["tier_performance"] = tp

    shared = {}
    for key in ("base_event_rate", "adas_usage", "clip_payload", "adas_multiplier"):
        if key not in raw_in:
            raise ValueError(f"INPUTS is missing '{key}'.")
        shared[key] = _two_tuple(raw_in[key], f"INPUTS['{key}']")

    telemetry = {}
    for group in TELEMETRY_GROUPS:
        if group not in raw_in:
            raise ValueError(f"INPUTS is missing telemetry group '{group}'.")
        v = raw_in[group]
        if not (isinstance(v, tuple) and len(v) == 4):
            raise ValueError(
                f"INPUTS['{group}'] must be a 4-tuple "
                f"(size_min, size_max, rate_min, rate_max), got {v!r}."
            )
        sz_lo, sz_hi, rt_lo, rt_hi = (float(x) for x in v)
        if sz_lo > sz_hi:
            raise ValueError(f"INPUTS['{group}']: size min > max.")
        if rt_lo > rt_hi:
            raise ValueError(f"INPUTS['{group}']: rate min > max.")
        telemetry[group] = (sz_lo, sz_hi, rt_lo, rt_hi)

    # ── MIXTURES ─────────────────────────────────────────────
    raw_mix = parsed["MIXTURES"]
    if not isinstance(raw_mix, dict):
        raise ValueError("MIXTURES must be a dict.")
    mixtures = {}
    for name in MIXTURE_LABELS:
        spec = raw_mix.get(name)
        if not isinstance(spec, dict) or any(k not in spec for k in MIXTURE_KEYS):
            raise ValueError(f"MIXTURES['{name}'] must be a dict with keys {', '.join(MIXTURE_KEYS)}.")
        m = {k: float(spec[k]) for k in MIXTURE_KEYS}
        if not 0.0 <= m["w"] <= 1.0:
            raise ValueError(f"MIXTURES['{name}']: w must be in [0, 1].")
        if min(m["a1"], m["b1"], m["a2"], m["b2"], m["scale"]) <= 0:
            raise ValueError(f"MIXTURES['{name}']: a1, b1, a2, b2 and scale must be > 0.")
        mixtures[name] = m

    # ── SPEED_DIST ───────────────────────────────────────────
    raw_tiers = parsed["SPEED_DIST"]
    if not isinstance(raw_tiers, dict) or not raw_tiers:
        raise ValueError("SPEED_DIST must be a non-empty dict.")

    tiers = {}
    for name, spec in raw_tiers.items():
        if not isinstance(spec, dict) or "upload_speed_mbps" not in spec or "weight" not in spec:
            raise ValueError(f"Tier '{name}' must have 'upload_speed_mbps' and 'weight'.")
        tiers[name] = {
            "upload_speed_mbps": float(spec["upload_speed_mbps"]),
            "weight":            float(spec["weight"]),
        }

    tier_w_sum = sum(t["weight"] for t in tiers.values())
    if abs(tier_w_sum - 1.0) > 1e-6:
        raise ValueError(f"SPEED_DIST weights must sum to 1.0 (currently {tier_w_sum:.4f}).")

    # ── CONGESTION ───────────────────────────────────────────
    raw_bw = parsed["CONGESTION"]
    if not isinstance(raw_bw, dict) or not raw_bw:
        raise ValueError("CONGESTION must be a non-empty dict.")
    bandwidth = {str(k): float(v) for k, v in raw_bw.items()}

    # ── PEAK_HOURS ───────────────────────────────────────────
    ph = parsed["PEAK_HOURS"]
    if not (isinstance(ph, tuple) and len(ph) == 2):
        raise ValueError(f"PEAK_HOURS must be a (start, end) 2-tuple, got {ph!r}.")
    p0, p1 = int(ph[0]), int(ph[1])
    if not (0 <= p0 <= p1 <= 24):
        raise ValueError("PEAK_HOURS must satisfy 0 <= start <= end <= 24.")

    return {
        "behaviour":             behaviour,
        "mixtures":              mixtures,
        "shared_ranges":         shared,
        "telemetry_groups":      telemetry,
        "nbn_tiers":             tiers,
        "bandwidth_utilisation": bandwidth,
        "peak_hours":            (p0, p1),
    }


def render_dict_input_tab():
    st.caption(
        "Paste a Python dict literal matching the notebook's structure: "
        "`INPUTS`, `MIXTURES`, `SPEED_DIST`, `CONGESTION`, `PEAK_HOURS`. "
        "Tier weights must sum to exactly 1.0. "
        "Loading here overwrites the editors in the other tabs."
    )

    if "dict_input_raw" not in st.session_state:
        st.session_state["dict_input_raw"] = _DEFAULT_DICT_STR

    raw = st.text_area(
        "Parameter dictionary",
        height=420,
        key="dict_input_raw",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        validate_clicked = st.button("Validate", use_container_width=True, key="dict_validate_btn")
    with c2:
        load_clicked = st.button("Load into config", type="primary",
                                 use_container_width=True, key="dict_load_btn")

    if validate_clicked or load_clicked:
        try:
            new_config = _parse_dict_to_config(raw)
            st.success("✅ Dictionary is valid.")
            if load_clicked:
                st.session_state["config"] = new_config
                st.success("✅ Loaded into config (switch tabs to see the values, or run the simulation).")
        except ValueError as e:
            st.error(f"❌ {e}")


def render_nbn_tier_editor():
    st.caption("Each tier has an upload speed (Mbps) and a population weight.")
    tiers = cfg()["nbn_tiers"]
    to_delete = None

    for name in list(tiers.keys()):
        t = tiers[name]
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 1])
            new_name = cols[0].text_input("Tier name", value=name, key=f"tier_name_{name}")
            t["upload_speed_mbps"] = cols[1].number_input(
                "Upload speed (Mbps)", min_value=0.1, value=float(t["upload_speed_mbps"]),
                step=1.0, key=f"tier_speed_{name}",
            )
            t["weight"] = cols[2].number_input(
                "Population weight", min_value=0.0, value=float(t["weight"]),
                step=0.01, key=f"tier_weight_{name}",
            )
            if cols[3].button("Remove", key=f"tier_del_{name}", use_container_width=True):
                to_delete = name

            if new_name != name and new_name.strip():
                if new_name in tiers:
                    st.error(f"A tier named '{new_name}' already exists.")
                else:
                    tiers[new_name] = tiers.pop(name)

    if to_delete:
        del tiers[to_delete]
        st.rerun()

    if st.button("➕ Add NBN tier", use_container_width=True, key="add_tier_btn"):
        base = "New Tier"
        new_name, i = base, 1
        while new_name in tiers:
            new_name = f"{base} {i}"
            i += 1
        tiers[new_name] = {"upload_speed_mbps": 25.0, "weight": 0.1}
        st.rerun()

    total_w = sum(t["weight"] for t in tiers.values())
    if tiers:
        ok = abs(total_w - 1.0) < 1e-6
        (st.caption if ok else st.error)(
            f"Total weight: {total_w:.4f}  "
            + ("✓" if ok else "(weights must sum exactly to 1.0)")
        )


def render_scenario_editor():
    st.caption(
        "A scenario scales every NBN tier's upload speed by a factor "
        "(e.g. 0.90 = 10% reduction from congestion). The factor applies over the whole "
        "transfer in the per-tier upload times, and only during the peak hours below in the "
        "busy-hour profile."
    )
    cfg()["peak_hours"] = tuple(st.slider(
        "Peak hours (local clock; start inclusive, end exclusive)", 0, 24,
        tuple(int(x) for x in cfg()["peak_hours"]), step=1, key="peak_hours_slider",
    ))

    scenarios = cfg()["bandwidth_utilisation"]
    to_delete = None

    for name in list(scenarios.keys()):
        with st.container(border=True):
            cols = st.columns([3, 2, 1])
            new_name = cols[0].text_input("Scenario name", value=name, key=f"scen_name_{name}")
            new_val = cols[1].number_input(
                "Utilisation factor", min_value=0.01, max_value=2.0,
                value=float(scenarios[name]), step=0.05, key=f"scen_val_{name}",
            )
            if cols[2].button("Remove", key=f"scen_del_{name}", use_container_width=True):
                to_delete = name
                continue
            scenarios[name] = new_val
            if new_name != name and new_name.strip():
                if new_name in scenarios:
                    st.error(f"A scenario named '{new_name}' already exists.")
                else:
                    scenarios[new_name] = scenarios.pop(name)

    if to_delete:
        del scenarios[to_delete]
        st.rerun()

    if st.button("➕ Add scenario", use_container_width=True, key="add_scenario_btn"):
        base = "New Scenario"
        new_name, i = base, 1
        while new_name in scenarios:
            new_name = f"{base} {i}"
            i += 1
        scenarios[new_name] = 1.0
        st.rerun()


# ─────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────

def render_results(results, profile, scenario, n, elapsed, config):
    upstream = results["upstream_gb"]
    up_stats = _stats(upstream)
    p0, p1   = config["peak_hours"]

    st.caption(
        f"Completed {n:,} simulations in {elapsed:.2f}s  ·  Scenario: **{scenario}**  ·  "
        f"Peak hours: **{_to_twelve_hour(p0)}–{_to_twelve_hour(p1)}**"
    )
    st.markdown("---")

    st.subheader("Results Summary", anchor=False)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean (GB)",            f"{up_stats['mean']:.3f}")
    k2.metric("Median (GB)",          f"{up_stats['median']:.3f}")
    k3.metric("P5 (GB)",              f"{up_stats['p5']:.3f}")
    k4.metric("P95 (GB)",             f"{up_stats['p95']:.3f}")
    k5.metric("Total (GB)", f"{upstream.sum():,.1f}")

    st.markdown("---")

    st.subheader("Upstream Data Distribution", anchor=False)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=upstream, nbinsx=60,
        marker_color=BAR_COLOR, marker_line_color="black",
        marker_line_width=0.4, opacity=0.82,
    ))
    fig_hist.add_vline(x=up_stats["mean"],   line_dash="dash", line_color="red",
                       annotation_text=f"Mean: {up_stats['mean']:.3f}",
                       annotation_position="top left", annotation_font_color="red")
    fig_hist.update_layout(
        xaxis=dict(title="Upstream GB / day", showgrid=False),
        yaxis=dict(title="Frequency", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    st.subheader("Upload Time by NBN Tier", anchor=False)
    st.caption("Payload ÷ (tier speed × scenario factor), applied over the whole transfer.")
    tier_cols  = [c for c in results.columns if c.endswith("_upload_hours")]
    tier_names = [c.replace("_upload_hours", "") for c in tier_cols]
    palette    = (TIER_PALETTE * ((len(tier_names) // len(TIER_PALETTE)) + 1))[:len(tier_names)]

    fig_box = go.Figure()
    for col, name, color in zip(tier_cols, tier_names, palette):
        fig_box.add_trace(go.Box(
            y=results[col], name=name,
            marker_color=color, boxmean=True, line_width=1.5,
        ))
    fig_box.update_layout(
        yaxis=dict(title="Upload time (hours)", gridcolor="lightgrey"),
        xaxis=dict(title="NBN Tier"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=420,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("NBN Tier Distribution", anchor=False)
    tier_order  = list(config["nbn_tiers"].keys())
    tier_counts = results["speed_tier"].value_counts().reindex(tier_order, fill_value=0)
    pct = 100 * tier_counts / len(results)

    fig_tier_pop = go.Figure()
    fig_tier_pop.add_trace(go.Bar(
        x=tier_order, y=tier_counts.values,
        marker_color=palette, marker_line_color="black", marker_line_width=0.6,
        text=[f"{p:.1f}%" for p in pct.values], textposition="outside",
    ))
    fig_tier_pop.update_layout(
        xaxis=dict(title="NBN Tier"),
        yaxis=dict(title="Number of users", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=40, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_tier_pop, use_container_width=True)

    st.markdown("---")

    st.subheader("Home Arrival Time Distribution", anchor=False)
    arrival = results["home_arrival_time"] % 24
    arr_mean = arrival.mean()

    fig_arrival = go.Figure()
    fig_arrival.add_trace(go.Histogram(
        x=arrival, xbins=dict(start=0, end=24, size=0.5),
        marker_color="steelblue", marker_line_color="black",
        marker_line_width=0.4, opacity=0.78,
    ))
    fig_arrival.add_vline(x=arr_mean,   line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {_to_twelve_hour(arr_mean)}",
                          annotation_position="top left", annotation_font_color="red")
    tick_h = list(range(0, 25, 2))
    fig_arrival.update_layout(
        xaxis=dict(title="Time of day", tickmode="array", tickvals=tick_h,
                   ticktext=[_to_twelve_hour(h) for h in tick_h], tickangle=-45),
        yaxis=dict(title="Number of users", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=30, b=70, l=60, r=30),
    )
    st.plotly_chart(fig_arrival, use_container_width=True)

    st.markdown("---")

    # Window shown on the time axes: at least 30 h, extended if uploads spill later
    last_active = int(profile.loc[profile["ev_count"] > 0, "hour"].max()) if (profile["ev_count"] > 0).any() else 0
    x_end = int(min(PROFILE_HOURS, max(30, last_active + 2)))
    x_end += x_end % 2

    st.subheader("Upload Start and End Times", anchor=False)
    st.caption("End time comes from the busy-hour simulation (tier performance and peak-hour throttling included).")
    start = results["home_arrival_time"].to_numpy()
    end   = results["upload_end_time"].to_numpy()
    tick_h2   = list(range(0, x_end + 1, 2))
    bin_edges = dict(start=0, end=x_end, size=0.5)

    col_s, col_e = st.columns(2)
    for col, vals, title, color in [
        (col_s, start, "Upload Start Time", "steelblue"),
        (col_e, end,   "Upload End Time",   BAR_COLOR),
    ]:
        mean_v = vals.mean()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, xbins=bin_edges,
            marker_color=color, marker_line_color="black",
            marker_line_width=0.4, opacity=0.78,
        ))
        fig.add_vline(x=mean_v,   line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {_to_twelve_hour(mean_v)}",
                      annotation_position="top left", annotation_font_color="red")
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font_size=14),
            xaxis=dict(title="Time of day", tickmode="array", tickvals=tick_h2,
                       ticktext=[_to_twelve_hour(h) for h in tick_h2], tickangle=-45),
            yaxis=dict(title="Number of users", gridcolor="lightgrey"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=380,
            margin=dict(t=50, b=70, l=60, r=30),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Network Busy Hour Profile", anchor=False)
    sl      = profile[profile["hour"] <= x_end]
    tick_h3 = list(range(0, x_end + 1, 2))

    col_ev, col_bw = st.columns(2)
    for col, y, ylabel, title, color, unit in [
        (col_ev, "ev_count",       "Number of EVs uploading",   "Active EVs per Hour",                 "steelblue", ""),
        (col_bw, "bandwidth_mbps", "Aggregate bandwidth (Mbps)", "Aggregate Upload Bandwidth per Hour", BAR_COLOR,   " Mbps"),
    ]:
        vals   = sl[y].values
        hrs    = sl["hour"].values
        peak_h = hrs[np.argmax(vals)]
        peak_v = vals.max()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hrs, y=vals, marker_color=color, marker_line_color="black",
            marker_line_width=0.4, opacity=0.82,
        ))
        fig.add_vline(x=peak_h, line_dash="dash", line_color="red",
                      annotation_text=f"Peak: {peak_v:.0f}{unit} at {_to_twelve_hour(peak_h)}",
                      annotation_position="top left", annotation_font_color="red")
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font_size=14),
            xaxis=dict(title="Hour of day", tickmode="array", tickvals=tick_h3,
                       ticktext=[_to_twelve_hour(h) for h in tick_h3], tickangle=-45),
            yaxis=dict(title=ylabel, gridcolor="lightgrey"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=380,
            margin=dict(t=50, b=70, l=60, r=30),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Export", anchor=False)
    st.download_button(
        "Download simulation results as CSV",
        data=results.to_csv(index=False),
        file_name=f"mc_{scenario.replace(' ', '_')}_n{n}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────
st.title("Monte Carlo Simulation", anchor=False)
st.markdown(
    "Simulate daily per-EV upstream data usage and upload time across the NBN tier mix. "
    "Every parameter below is editable. Add or remove NBN tiers and "
    "congestion scenarios as needed."
)

with st.expander("Simulation Input", expanded=True):
    st.markdown("##### Run settings")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        n = st.number_input("Number of simulations", value=10000, min_value=0, step=1000, key="run_n")
    with fc2:
        scenario_options = list(cfg()["bandwidth_utilisation"].keys())
        scenario = st.selectbox("Scenario", scenario_options, key="run_scenario") if scenario_options else None
    with fc3:
        seed = st.number_input("Random seed", value=42, min_value=0, step=1, key="run_seed")

    st.markdown("---")
    st.markdown("##### Choose Input Method")
    input_mode = st.radio(
        label="Available methods",
        options=["Sliders", "Manual Dictionary Input"],
        horizontal=True,
        key="top_input_mode",
    )
    st.markdown("---")

    if input_mode == "Sliders":
        sub_tabs = st.tabs(["Driver Behaviour", "NBN Tiers", "Congestion Scenarios", "Advanced Parameters"])
        with sub_tabs[0]:
            render_behaviour_editor()
        with sub_tabs[1]:
            render_nbn_tier_editor()
        with sub_tabs[2]:
            render_scenario_editor()
        with sub_tabs[3]:
            render_advanced_editor()
    else:
        render_dict_input_tab()

    st.markdown("---")
    run_clicked = st.button("▶  Run simulation", type="primary", use_container_width=True, key="run_btn")


if run_clicked:
    errors = []
    if not cfg()["nbn_tiers"]:
        errors.append("At least one NBN tier is required.")
    if not scenario:
        errors.append("At least one bandwidth scenario is required.")
    if not n:
        errors.append("Number of simulations must be at least 1.")

    tier_w_sum = sum(t["weight"] for t in cfg()["nbn_tiers"].values())
    if cfg()["nbn_tiers"] and abs(tier_w_sum - 1.0) > 1e-6:
        errors.append(f"NBN tier weights must sum to 1.0 (currently {tier_w_sum:.4f}).")

    if cfg()["peak_hours"][0] >= cfg()["peak_hours"][1]:
        errors.append("Peak hours: start must be before end.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        config_snapshot = copy.deepcopy(cfg())
        cache_key = _config_cache_key(config_snapshot, n, scenario, seed)
        with st.spinner(f"Running {n:,} simulations…"):
            t0 = time.perf_counter()
            results = run_simulation(n, scenario, seed, config_snapshot, cache_key)
            profile, finish = compute_hourly_profile(results, scenario, config_snapshot)
            results = results.assign(upload_end_time=finish)
            st.session_state["mc_results"]       = results
            st.session_state["mc_profile"]       = profile
            st.session_state["mc_elapsed"]       = time.perf_counter() - t0
            st.session_state["mc_scenario"]      = scenario
            st.session_state["mc_n"]             = n
            st.session_state["mc_run"]           = True
            st.session_state["mc_config_used"]   = config_snapshot

if st.session_state["mc_run"] and st.session_state["mc_results"] is not None:
    render_results(
        st.session_state["mc_results"],
        st.session_state["mc_profile"],
        st.session_state["mc_scenario"],
        st.session_state["mc_n"],
        st.session_state["mc_elapsed"],
        st.session_state["mc_config_used"],
    )