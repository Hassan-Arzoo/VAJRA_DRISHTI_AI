"""Interactive synthetic storm-monitoring console for the VajraDrishti PoC."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from vajradrishti.model import HORIZONS, train_model
from vajradrishti.pipeline import run_prediction
from vajradrishti.synthetic import SOURCES, demo_replay


st.set_page_config(page_title="VajraDrishti | Storm Console", page_icon="⛈️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#edf3ff; --muted:#9ba9c4; --teal:#61e3d0; --amber:#ffc477; --violet:#b4a3ff; --line:rgba(157,177,222,.17); }
.stApp { background: radial-gradient(ellipse at 82% 2%, rgba(54,91,174,.27), transparent 36%), radial-gradient(ellipse at 8% 35%, rgba(125,61,145,.20), transparent 34%), #080f20; color:var(--ink); }
.block-container { max-width:1480px; padding-top:5.25rem; padding-bottom:2.5rem; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.topline { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding:4px 0 15px; margin-bottom:18px; }
.brand { display:flex; gap:12px; align-items:center; font-weight:800; letter-spacing:.015em; font-size:1rem; }
.brand-title { font-size:clamp(.82rem,1.45vw,1.08rem); line-height:1.3; }
.brandmark { display:grid; place-items:center; width:37px; height:37px; border:1px solid rgba(97,227,208,.38); border-radius:12px; background:linear-gradient(145deg,#1d4161,#29234c); color:var(--teal); font-size:1.15rem; }
.topmeta { color:var(--muted); font:500 .69rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
.team-name { color:var(--teal); font-weight:700; text-transform:none; }
.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:22px; padding:27px 30px; margin-bottom:15px; background:linear-gradient(112deg,rgba(18,44,78,.97),rgba(20,31,57,.96) 58%,rgba(48,34,70,.88)); box-shadow:0 18px 55px rgba(1,6,20,.22); }
.hero:after { content:'⛈'; position:absolute; right:6%; top:-48px; font-size:190px; opacity:.065; filter:grayscale(1); pointer-events:none; }
.eyebrow { color:var(--teal); font:500 .68rem 'DM Mono',monospace; letter-spacing:.15em; text-transform:uppercase; }
.hero h1 { margin:.45rem 0 .25rem; font-size:clamp(1.7rem,3vw,2.5rem); line-height:1.13; letter-spacing:-.045em; color:#eff9f7; }
.hero p { margin:0; max-width:720px; color:#b4c3dc; font-size:.92rem; }
.livepill { display:inline-flex; align-items:center; gap:8px; border:1px solid rgba(255,189,105,.24); border-radius:999px; padding:7px 11px; color:#ffd19b; background:rgba(117,70,31,.16); font:500 .66rem 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
.dot { width:7px; height:7px; background:var(--amber); border-radius:50%; box-shadow:0 0 12px var(--amber); }
.section-label { color:#789396; font:500 .68rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; margin:4px 0 10px; }
div[data-testid="stMetric"] { min-height:123px; padding:18px 20px; border:1px solid var(--line); border-radius:17px; background:linear-gradient(145deg,rgba(23,42,75,.94),rgba(15,27,52,.96)); box-shadow:0 12px 30px rgba(0,0,0,.17); }
div[data-testid="stMetricLabel"] { color:#aab9d3; font-size:.77rem; }
div[data-testid="stMetricValue"] { color:#f2f6ff; font-weight:700; letter-spacing:-.04em; }
div[data-testid="stMetricDelta"] { font:500 .7rem 'DM Mono',monospace; }
div[data-testid="stRadio"] label, div[data-testid="stSegmentedControl"] label { color:#b8c9c9; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:18px; background:rgba(13,27,30,.62); }
.availability { border:1px solid var(--line); border-radius:17px; background:rgba(15,28,53,.76); padding:18px 19px; }
.source-row { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:9px 0; border-bottom:1px solid rgba(148,190,191,.09); color:#c5d3d3; font-size:.8rem; }
.source-row:last-child { border:0; padding-bottom:0; }
.source-state { font:500 .64rem 'DM Mono',monospace; letter-spacing:.07em; text-transform:uppercase; white-space:nowrap; }
.good { color:#70e3c4; } .bad { color:#ff9da9; }
.crew-row { display:flex; flex-wrap:wrap; align-items:center; gap:7px; color:#8da6aa; font-size:.8rem; }
.team-person { display:inline-block; padding:4px 8px; border-radius:8px; color:#d6e5e3; transition:color .2s ease, background .2s ease, box-shadow .2s ease, transform .2s ease, filter .2s ease; cursor:default; }
.team-person:hover { color:#c1fff4; background:linear-gradient(115deg,rgba(57,210,202,.19),rgba(145,119,255,.18)); box-shadow:0 0 20px rgba(97,227,208,.27), inset 0 0 0 1px rgba(180,163,255,.2); transform:translateY(-2px); filter:brightness(1.25); }
.crew-label { color:#7e999b; font:500 .62rem 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; margin-right:5px; }
.team-stack { display:flex; flex-direction:column; align-items:flex-start; gap:2px; }
.team-stack .team-person { margin-left:-8px; }
.risk-chip { display:inline-flex; align-items:center; border-radius:99px; padding:5px 10px; font:500 .65rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; background:rgba(255,189,105,.12); color:#ffd19b; border:1px solid rgba(255,189,105,.2); }
.footnote { border-top:1px solid var(--line); margin-top:24px; padding-top:13px; color:#73898c; font:400 .66rem 'DM Mono',monospace; line-height:1.7; }
button[kind="secondary"] { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return train_model()


def risk_band(probability: float) -> tuple[str, str]:
    if probability >= .75:
        return "Very high", "#fb7881"
    if probability >= .50:
        return "High", "#ffbd69"
    if probability >= .25:
        return "Moderate", "#f0d27a"
    return "Low", "#66d6b5"


def replay_track(events: list[dict], through: int) -> tuple[list[int], list[int]]:
    xs, ys = [], []
    for event in events[:through]:
        grid = np.asarray(event["radar"]["reflectivity_dbz"], dtype=float)
        if np.isfinite(grid).any():
            py, px = np.unravel_index(np.nanargmax(grid), grid.shape)
            xs.append(int(px)); ys.append(int(py))
    return xs, ys


st.markdown("""
<div class="topline">
  <div class="brand"><span class="brandmark">✳</span><span class="brand-title">VajraDrishti AI: Multimodal Thunderstorm &amp; Lightning Nowcasting System</span></div>
</div>
""", unsafe_allow_html=True)

weather_scene = Path(__file__).with_name("weather_scene.html").read_text(encoding="utf-8")
components.html(weather_scene, height=430, scrolling=False)

# Persistent controls keep the replay predictable across Streamlit reruns.
st.session_state.setdefault("replay_step", 7)
st.session_state.setdefault("autoplay", False)

control_a, control_b, control_c, control_d = st.columns([1.35, 1.15, 1.3, 1.3])
with control_a:
    st.markdown('<div class="section-label">01 / Event replay</div>', unsafe_allow_html=True)
    scenario = st.selectbox("Synthetic event", ["Synthetic replay A", "Synthetic replay B", "Synthetic replay C"], label_visibility="collapsed")
with control_b:
    st.markdown('<div class="section-label">02 / Missing sources</div>', unsafe_allow_html=True)
    missing = st.multiselect("Simulate unavailable", SOURCES, label_visibility="collapsed", placeholder="All sources online")
with control_c:
    st.markdown('<div class="section-label">03 / Map layer</div>', unsafe_allow_html=True)
    layer = st.selectbox("Grid layer", ["Fused activity", "Radar reflectivity", "Cloud index", "Lightning density"], label_visibility="collapsed")
with control_d:
    st.markdown('<div class="section-label">04 / Display</div>', unsafe_allow_html=True)
    show_track = st.toggle("Show storm-core trail", value=True)

scenario_seed = {"Synthetic replay A": 42, "Synthetic replay B": 117, "Synthetic replay C": 26072}[scenario]
events = demo_replay(seed=scenario_seed, missing=tuple(missing))

play_a, play_b, play_c, play_d = st.columns([.6, .6, 4.2, 1.1])
with play_a:
    if st.button("◀", help="Previous replay scene", use_container_width=True):
        st.session_state.replay_step = max(1, st.session_state.replay_step - 1)
with play_b:
    if st.button("▶", help="Advance one replay scene", use_container_width=True):
        st.session_state.replay_step = st.session_state.replay_step % len(events) + 1
with play_c:
    step = st.slider("Storm replay scene", 1, len(events), st.session_state.replay_step, label_visibility="collapsed", format="Scene %d / 13")
    st.session_state.replay_step = step
with play_d:
    st.markdown(f'<div class="livepill"><span class="dot"></span> DEMO SCENE {step:02d} / 13</div>', unsafe_allow_html=True)

payload = events[step - 1]
result = run_prediction(payload, get_models())
horizon = st.pills("Forecast lead time", HORIZONS, format_func=lambda v: f"+{v:02d} MIN", default=10, selection_mode="single")
horizon = horizon or 10
key = f"{horizon}_min"
lightning_p = result["predictions"]["lightning_probability"][key]
storm_p = result["predictions"]["thunderstorm_probability"][key]
peak_p = max(lightning_p, storm_p)
risk, risk_color = risk_band(peak_p)

k1, k2, k3, k4 = st.columns([1, 1, .9, 1])
k1.metric("⚡  Lightning probability", f"{lightning_p:.0%}", f"+{horizon} minute outlook")
k2.metric("⛈  Thunderstorm hazard", f"{storm_p:.0%}", f"+{horizon} minute outlook")
k3.metric("Prototype risk band", risk, "Visualization only")
k4.metric("Source completeness", f"{sum(result['data_availability'].values())}/4", f"Availability index · {result['confidence']:.0%}")

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="section-label">Spatial view &nbsp; / &nbsp; simplified synthetic grid</div>', unsafe_allow_html=True)
    radar = np.asarray(payload["radar"]["reflectivity_dbz"], dtype=float)
    cloud = np.asarray(payload["satellite"]["cloud_index"], dtype=float)
    flashes = np.asarray(payload["lightning"]["flash_counts"], dtype=float)
    fields = {
        "Fused activity": (np.nan_to_num(radar / 75) * .55 + np.nan_to_num(cloud) * .30 + np.nan_to_num(flashes / 9) * .15, "Relative activity"),
        "Radar reflectivity": (radar, "Synthetic reflectivity (dBZ-like)"),
        "Cloud index": (cloud, "Synthetic cloud index"),
        "Lightning density": (flashes, "Synthetic flash count / cell"),
    }
    z, scale_label = fields[layer]
    colors = [[0, "#111a33"], [.22, "#39457f"], [.5, "#24aaa8"], [.74, "#f4c266"], [1, "#f0798c"]]
    fig = go.Figure(go.Heatmap(z=z, colorscale=colors, colorbar={"title": {"text": scale_label, "side": "right"}, "thickness": 12, "outlinewidth": 0}, hovertemplate="Grid cell (%{x}, %{y})<br>Value: %{z:.2f}<extra></extra>"))
    if show_track:
        tx, ty = replay_track(events, step)
        if len(tx) > 1:
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines+markers", line={"color":"#fff0c4", "width":2, "dash":"dot"}, marker={"color":"#fff0c4", "size":7}, name="Storm-core trail", hovertemplate="Replay core · (%{x}, %{y})<extra></extra>"))
    ly, lx = np.where(np.nan_to_num(flashes) > 3)
    if layer != "Lightning density" and len(lx):
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers", marker={"size":6, "color":"#fff2c5", "symbol":"circle-open", "line":{"width":1.4}}, name="Lightning cells", hoverinfo="skip"))
    fig.update_layout(height=430, margin={"l":5,"r":5,"t":12,"b":5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0c1428", font={"color":"#a6b3d0","family":"DM Mono, monospace","size":10}, xaxis={"title":"GRID COLUMN","showgrid":False,"zeroline":False,"tickfont":{"size":9}}, yaxis={"title":"GRID ROW","showgrid":False,"zeroline":False,"autorange":"reversed","tickfont":{"size":9},"scaleanchor":"x"}, legend={"orientation":"h","y":1.03,"x":0,"bgcolor":"rgba(0,0,0,0)"})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})

    # Compact lead-time sparkline stays tied to the horizon selector above.
    st.markdown('<div class="section-label">Probability timeline &nbsp; / &nbsp; every forecast lead</div>', unsafe_allow_html=True)
    horizons = [f"{v}_min" for v in HORIZONS]
    line = go.Figure()
    line.add_trace(go.Scatter(x=[5,10,15,30,60], y=[result["predictions"]["lightning_probability"][k] for k in horizons], mode="lines+markers", name="Lightning", line={"color":"#ffc477","width":2.5}, marker={"size":7}))
    line.add_trace(go.Scatter(x=[5,10,15,30,60], y=[result["predictions"]["thunderstorm_probability"][k] for k in horizons], mode="lines+markers", name="Thunderstorm", line={"color":"#b4a3ff","width":2.5}, marker={"size":7}))
    line.add_vline(x=horizon, line_dash="dot", line_color="rgba(230,241,236,.5)")
    line.update_layout(height=180, margin={"l":0,"r":4,"t":8,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color":"#82999b","family":"DM Mono, monospace","size":9}, xaxis={"title":"MINUTES AHEAD","tickvals":[5,10,15,30,60],"gridcolor":"rgba(148,190,191,.08)","zeroline":False}, yaxis={"title":"PROBABILITY","range":[0,1],"tickformat":".0%","gridcolor":"rgba(148,190,191,.08)","zeroline":False}, legend={"orientation":"h","y":1.15,"x":0,"bgcolor":"rgba(0,0,0,0)"})
    st.plotly_chart(line, width="stretch", config={"displayModeBar":False})

with right:
    st.markdown('<div class="section-label">Observation channels &nbsp; / &nbsp; source health</div>', unsafe_allow_html=True)
    names = {"radar":"Radar-like reflectivity", "satellite":"Satellite-like cloud", "lightning":"Lightning observations", "atmosphere":"Atmospheric context"}
    rows = []
    for source, label in names.items():
        available = result["data_availability"][source]
        state = "AVAILABLE" if available else "SIMULATED OFFLINE"
        style = "good" if available else "bad"
        glyph = "●" if available else "○"
        rows.append(f'<div class="source-row"><span>{glyph} &nbsp; {label}</span><span class="source-state {style}">{state}</span></div>')
    st.markdown('<div class="availability">' + "".join(rows) + '</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div><div class="section-label">Selected horizon &nbsp; / &nbsp; forecast readout</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="border:1px solid {risk_color}45;border-radius:18px;padding:20px;background:linear-gradient(135deg,{risk_color}13,rgba(13,27,30,.8));"><div style="font:500 .68rem DM Mono,monospace;color:#9fb7b9;letter-spacing:.1em">{horizon} MINUTES AHEAD</div><div style="font-size:2.2rem;font-weight:800;letter-spacing:-.05em;color:{risk_color};margin:4px 0 2px">{risk.upper()}</div><div style="color:#91a8aa;font-size:.76rem">Highest of the two displayed demo probabilities</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Replay context</div>', unsafe_allow_html=True)
    st.caption(f"Synthetic event: **{payload['event_id']}**  \nScene **{step} of {len(events)}** · {payload['timestamp']}")
    with st.expander("How to read this prototype"):
        st.write("The colored grid is a simplified synthetic activity field, not a geographic map. Probabilities come from a Random Forest trained on synthetic proxy labels. Source completeness is not model accuracy or statistical confidence.")

st.markdown('<div class="footnote">SYNTHETIC / DEMO OUTPUT &nbsp;·&nbsp; Prototype risk bands are for interface demonstration only, not official warning thresholds. No operational meteorological accuracy is claimed.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label" style="margin-top:28px">Project team</div>', unsafe_allow_html=True)
with st.container(border=True):
    roster_left, roster_right = st.columns([1.1, 4])
    with roster_left:
        st.markdown('<div class="team-stack"><span class="team-person"><strong>© Team ZeninClan</strong></span><span class="team-person">Aliah University, Kolkata</span><span class="team-person">SIH 2026 · PS 26072</span></div>', unsafe_allow_html=True)
    with roster_right:
        st.markdown('<div class="crew-row"><span class="crew-label">Team Lead</span><span class="team-person">Shafaqun Nisa</span></div><div class="crew-row"><span class="crew-label">Members</span><span class="team-person">Nishat Khanam</span><span class="team-person">Huma Mehfooz</span><span class="team-person">Rafaquat Raza</span><span class="team-person">Md Akhmal Alam</span><span class="team-person">Askala Perveen</span></div>', unsafe_allow_html=True)
