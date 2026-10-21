"""Shared look and feel: colours, strategy names, CSS, data loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "results" / "precomputed"

INK = "#13233A"
MUTED = "#5B6B7F"
TEAL = "#0B7A6F"
TEAL_LIGHT = "#2BB3A3"
RED = "#D9480F"
VIOLET = "#6C5DD3"
GREY = "#8C96A3"
BASE = "#B7C0CB"
YELLOW = "#F2B705"
FONT = "IBM Plex Sans, system-ui, sans-serif"

STRATEGY = {
    "fcfs": {"name": "Charge on arrival", "long": "Charge on arrival (today)", "color": RED, "dash": "solid"},
    "smart": {"name": "Smart apps, uncoordinated", "long": "Each driver's app picks their cheapest slot",
              "color": VIOLET, "dash": "dot"},
    "quantum": {"name": "Quantum-coordinated", "long": "Decomposed XY-QAOA (this project)",
                "color": TEAL, "dash": "solid"},
    "exact": {"name": "Classical exact", "long": "Same decomposition, brute force (reference)",
              "color": GREY, "dash": "dash"},
}


def inject_css():
    st.markdown(
        """
<style>
.block-container {padding-top: 2.2rem; max-width: 1240px;}
h1 {letter-spacing: -0.01em; line-height: 1.08;}
.hero-title {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 700;
  font-size: clamp(2.0rem, 4.2vw, 3.3rem); line-height: 1.05; margin: 0 0 .6rem 0; max-width: 22ch;}
.hero-title .num-bad {color: #D9480F;}
.hero-title .num-good {color: #0B7A6F;}
.lede {font-size: 1.12rem; max-width: 64ch; opacity: .86; margin-bottom: 1.2rem;}
.step-n {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 700; font-size: 2.1rem;
  color: #0B7A6F; line-height: 1; margin-bottom: .35rem;}
.step-t {font-weight: 600; margin-bottom: .2rem;}
.step-d {font-size: .93rem; opacity: .82;}
.plan-hub {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 700; font-size: 2.3rem;
  line-height: 1.05; margin: .1rem 0 .4rem 0;}
.note {font-size: .85rem; opacity: .72;}
.chip {display: inline-block; padding: .1rem .55rem; border-radius: 1rem; font-size: .82rem;
  background: rgba(242,183,5,.22); margin-right: .3rem;}
div[data-testid="stMetricValue"] {font-family: "IBM Plex Sans Condensed", sans-serif;}
@media (prefers-reduced-motion: reduce) { * {animation: none !important; transition: none !important;} }
</style>
""",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 12, b=70 if (title and legend) else 8),
        font=dict(family=FONT, size=13),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        hovermode="x unified",
        showlegend=legend,
        legend=(dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title=None) if title
                else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(140,150,163,.25)", zeroline=False)
    return fig


def show(fig, key=None, modebar=True):
    config = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
    if not modebar:
        config["displayModeBar"] = False
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config=config)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    path = PRE / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def hour_label(h: int) -> str:
    return f"{h:02d}:00"


def rs(x: float) -> str:
    return f"₹{x:,.0f}"
