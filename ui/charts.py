"""Plotly chart builders. Every function takes plain data (dicts / lists) so the same
charts render precomputed results and live runs."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import BASE, GREY, INK, MUTED, RED, STRATEGY, TEAL, VIOLET, YELLOW, hour_label, style_fig

HOURS = list(range(24))
MAP_STYLE = "carto-positron"
HOUR_LABELS = [hour_label(h) for h in HOURS]


def _solar_band(fig, city=None, row=None, col=None, label=True):
    lo, hi = (9, 16) if city is None else city["meta"]["solar_hours"]
    kw = dict(row=row, col=col) if row else {}
    fig.add_vrect(x0=lo - 0.5, x1=hi - 0.5, fillcolor=YELLOW, opacity=0.13, line_width=0, layer="below", **kw)
    if label:
        fig.add_annotation(x=lo - 0.4, y=0.02, yref="paper", xanchor="left", yanchor="bottom",
                           text="Solar hours, 0.7× tariff", showarrow=False, font=dict(size=11, color=MUTED), **kw)


# ------------------------------------------------------------------ city

def city_load_fig(city, strategies=("fcfs", "smart", "quantum"), height=420, hour=None, title=None, compact=False):
    base = np.array(city["base_actual"]).sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=HOURS, y=base, name="Feeder load without EVs", mode="lines",
                             line=dict(color=BASE, width=0), fill="tozeroy", fillcolor="rgba(183,192,203,.45)",
                             hovertemplate="%{y:,.0f} kW"))
    for s in strategies:
        ev = np.array(city["strategies"][s]["ev_kw"]).sum(axis=1)
        meta = STRATEGY[s]
        fig.add_trace(go.Scatter(x=HOURS, y=base + ev, name=meta["name"], mode="lines",
                                 line=dict(color=meta["color"], width=3 if s == "quantum" else 2.2,
                                           dash=meta["dash"], shape="spline", smoothing=0.5),
                                 hovertemplate="%{y:,.0f} kW"))
    peak = float(base.max())
    fig.add_hline(y=peak, line=dict(color=INK, width=1, dash="dot"), opacity=0.5)
    fig.add_annotation(x=1, xref="paper", xanchor="right", y=peak, yanchor="bottom",
                       text=f"Evening peak without EVs: {peak:,.0f} kW", showarrow=False,
                       font=dict(size=11, color=MUTED))
    _solar_band(fig, city)
    if hour is not None:
        fig.add_vline(x=hour, line=dict(color=INK, width=1.5))
    step = 4 if compact else 3
    fig.update_xaxes(tickvals=HOURS[::step], ticktext=HOUR_LABELS[::step], range=[-0.5, 23.5], tickangle=0)
    fig.update_yaxes(title=None if compact else "City feeder load (kW)", rangemode="tozero")
    fig = style_fig(fig, height, title=title)
    if compact:
        fig.update_layout(legend=dict(font=dict(size=11)), margin=dict(t=56))
    return fig


def ev_load_fig(city, strategies=("fcfs", "smart", "quantum"), height=320):
    fig = go.Figure()
    for s in strategies:
        ev = np.array(city["strategies"][s]["ev_kw"]).sum(axis=1)
        meta = STRATEGY[s]
        fig.add_trace(go.Bar(x=HOURS, y=ev, name=meta["name"], marker_color=meta["color"],
                             hovertemplate="%{y:,.0f} kW"))
    _solar_band(fig, city, label=False)
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.05)
    fig.update_xaxes(tickvals=HOURS[::2], ticktext=HOUR_LABELS[::2])
    fig.update_yaxes(title="EV charging load (kW)")
    return style_fig(fig, height)


def hub_heatmap_fig(city, strategies=("fcfs", "quantum"), height=330):
    hubs = [h["name"] for h in city["hubs"]]
    fig = make_subplots(rows=1, cols=len(strategies), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[STRATEGY[s]["name"] for s in strategies])
    zmax = max(np.array(city["strategies"][s]["ev_kw"]).max() for s in strategies)
    for i, s in enumerate(strategies, start=1):
        z = np.array(city["strategies"][s]["ev_kw"]).T
        fig.add_trace(go.Heatmap(z=z, x=HOURS, y=hubs, zmin=0, zmax=zmax, coloraxis="coloraxis",
                                 hovertemplate="%{y}, %{x}:00<br>%{z:.0f} kW<extra></extra>"), row=1, col=i)
        fig.update_xaxes(tickvals=HOURS[::6], ticktext=HOUR_LABELS[::6], tickangle=0, row=1, col=i)
    fig.update_layout(coloraxis=dict(colorscale=[[0, "rgba(11,122,111,0.04)"], [0.25, "#9AD3CC"],
                                                 [0.6, "#2BB3A3"], [1, "#0A4F48"]],
                                     colorbar=dict(title="kW", thickness=10)))
    for a in fig.layout.annotations:
        a.update(font=dict(size=13), yshift=6)
    fig = style_fig(fig, height, legend=False)
    fig.update_layout(margin=dict(t=36))
    return fig


def city_map_fig(city, strategy="quantum", height=470, highlight=None, pref=None, animate=True, hour=18):
    hubs = city["hubs"]
    lat = [h["lat"] for h in hubs]
    lon = [h["lon"] for h in hubs]
    names = [h["name"] for h in hubs]
    ev = np.array(city["strategies"][strategy]["ev_kw"])
    ports = np.array(city["strategies"][strategy]["ports"])
    cap = np.array([h["ports"] for h in hubs])

    def frame_data(h):
        util = ports[h] / cap
        size = 14 + ev[h] * 0.55
        text = [f"<b>{n}</b><br>{hour_label(h)}: {ev[h, i]:.0f} kW EV load<br>{ports[h, i]} of {cap[i]} ports busy"
                for i, n in enumerate(names)]
        return go.Scattermap(lat=lat, lon=lon, mode="markers+text", text=names, textposition="top center",
                             textfont=dict(size=12, color=INK),
                             marker=dict(size=size, color=util, cmin=0, cmax=1, opacity=0.85,
                                         colorscale=[[0, "#CFE9E5"], [0.5, TEAL], [1, RED]],
                                         colorbar=dict(title="Ports busy", tickformat=".0%", thickness=10,
                                                       len=0.5, x=0.99, xanchor="right", y=0.97, yanchor="top",
                                                       bgcolor="rgba(255,255,255,.75)")),
                             hovertext=text, hoverinfo="text", name="Hubs")

    fig = go.Figure(frame_data(hour))
    if highlight is not None:
        fig.add_trace(go.Scattermap(lat=[lat[highlight]], lon=[lon[highlight]], mode="markers",
                                    marker=dict(size=46, color=YELLOW, opacity=0.55),
                                    hoverinfo="skip", name="Your charger"))
    if pref is not None and pref != highlight:
        fig.add_trace(go.Scattermap(lat=[lat[pref]], lon=[lon[pref]], mode="markers",
                                    marker=dict(size=18, color=INK, opacity=0.9),
                                    hoverinfo="skip", name="Where you are"))
    if animate:
        fig.frames = [go.Frame(data=[frame_data(h)], name=str(h), traces=[0]) for h in HOURS]
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.01, y=0.99, xanchor="left", yanchor="top",
                              pad=dict(t=0, r=4), showactive=False, bgcolor="rgba(255,255,255,.9)",
                              font=dict(color=INK, size=13),
                              buttons=[dict(label="▶ Play the day", method="animate",
                                            args=[None, dict(frame=dict(duration=650, redraw=True),
                                                             transition=dict(duration=250), fromcurrent=True)]),
                                       dict(label="❚❚", method="animate",
                                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                               mode="immediate")])])],
            sliders=[dict(active=hour, x=0.02, len=0.96, y=0.0, yanchor="top", pad=dict(t=6, b=0),
                          bgcolor="rgba(140,150,163,.35)", activebgcolor=TEAL, font=dict(size=11),
                          currentvalue=dict(prefix="Hour shown: ", font=dict(size=13)), ticklen=3,
                          steps=[dict(label=hour_label(h), method="animate",
                                      args=[[str(h)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                                 for h in HOURS])])
    fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=12.955, lon=77.655), zoom=10.3),
                      height=height, margin=dict(l=0, r=0, t=0, b=70 if animate else 0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def qaoa_engine_fig(city, height=300):
    log = city["strategies"]["quantum"].get("qaoa_log", [])
    if not log:
        return None
    idx = list(range(1, len(log) + 1))
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Bar(x=idx, y=[x["qubits"] for x in log], name="Qubits in sub-problem",
                         marker_color=[TEAL if x["found_optimum"] else YELLOW for x in log],
                         customdata=[[x["hour"], x["vehicles"], x["evaluations"], 100 * x["p_optimal"]] for x in log],
                         hovertemplate="Sub-problem %{x}<br>Hour %{customdata[0]}:00, %{customdata[1]} vehicles"
                                       "<br>%{y} qubits, %{customdata[2]} circuit evaluations"
                                       "<br>P(optimal) %{customdata[3]:.1f}%<extra></extra>"))
    fig.update_xaxes(title="Sub-problem, in the order the day was solved")
    fig.update_yaxes(title="Qubits", range=[0, 17])
    fig.update_layout(hovermode="closest")
    return style_fig(fig, height, legend=False)


# ------------------------------------------------------------------ quantum lab

def forecast_fig(fc, height=340):
    hist, tgt = fc["history"], fc["target"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["demand_kw"], name="Observed load",
                             line=dict(color=MUTED, width=1.6)))
    fig.add_trace(go.Scatter(x=tgt["timestamp"] + tgt["timestamp"][::-1], y=tgt["p90"] + tgt["p10"][::-1],
                             fill="toself", fillcolor="rgba(11,122,111,.15)", line=dict(width=0),
                             name="P10 to P90 band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tgt["timestamp"], y=tgt["actual"], name="Actual (held out)",
                             line=dict(color=MUTED, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=tgt["timestamp"], y=tgt["pred"], name="AI forecast", line=dict(color=TEAL, width=2.6)))
    hrs = fc["window_hours"]
    day = tgt["timestamp"][0][:10]
    fig.add_vrect(x0=f"{day} {hrs[0]:02d}:00", x1=f"{day} {hrs[-1]:02d}:59", fillcolor=YELLOW, opacity=0.18,
                  line_width=0, annotation_text="Scheduling window", annotation_position="top left")
    fig.update_yaxes(title="Feeder load (kW)")
    return style_fig(fig, height)


def slot_load_fig(lab, methods=("FCFS", "Quantum (XY-QAOA)"), height=330):
    inst = lab["instance"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=inst["slots"], y=inst["base_load"], name="Forecast base load", marker_color=BASE))
    colors = {"FCFS": RED, "Quantum (XY-QAOA)": TEAL, "Classical optimum": GREY}
    for m in methods:
        ev = lab["metrics"][m]["ev_load_kw"]
        fig.add_trace(go.Bar(x=inst["slots"], y=ev, name=f"EV load, {m}", marker_color=colors.get(m, VIOLET),
                             offsetgroup=m, base=inst["base_load"]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Total load (kW)")
    return style_fig(fig, height)


def convergence_fig(history, optimal, color=TEAL, height=300, title=None):
    fig = go.Figure(go.Scatter(y=history, mode="lines", line=dict(color=color, width=2), name="CVaR cost"))
    fig.add_hline(y=optimal, line=dict(color=GREY, dash="dash"), annotation_text="Exact optimum",
                  annotation_position="bottom right")
    fig.update_xaxes(title="Circuit evaluation")
    fig.update_yaxes(title="CVaR cost (₹)")
    return style_fig(fig, height, legend=False, title=title)


def distribution_fig(dist, height=330, title=None):
    colors = [TEAL if d["optimal"] else ("#7FC8BF" if d["valid"] else BASE) for d in dist]
    fig = go.Figure(go.Bar(x=[f"₹{d['cost']:.0f}" for d in dist], y=[100 * d["prob"] for d in dist],
                           marker_color=colors, showlegend=False,
                           customdata=[[d["bits"], "optimal" if d["optimal"] else ("valid" if d["valid"] else "breaks a rule")]
                                       for d in dist],
                           hovertemplate="%{customdata[0]}<br>cost %{x}, %{customdata[1]}<br>%{y:.2f}% of shots<extra></extra>"))
    fig.update_xaxes(title=None, type="category", tickangle=-60)
    fig.update_yaxes(title="Share of shots (%)")
    fig.update_layout(hovermode="closest")
    for name, c in [("Optimal schedule", TEAL), ("Valid schedule", "#7FC8BF"), ("Breaks a rule", BASE)]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=name))
    return style_fig(fig, height, title=title)


def mixer_fig(qaoa, random_p, height=300):
    names = list(qaoa)
    short = [n.replace("Quantum (", "").rstrip(")") for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=short,
                         x=[100 * qaoa[n]["summary"]["prob_feasible"] for n in names],
                         orientation="h", name="P(valid schedule)", marker_color="#7FC8BF"))
    fig.add_trace(go.Bar(y=short,
                         x=[100 * qaoa[n]["summary"]["prob_optimal"] for n in names],
                         orientation="h", name="P(optimal schedule)", marker_color=TEAL))
    fig.add_vline(x=100 * random_p, line=dict(color=RED, dash="dot"),
                  annotation_text="Random valid guess", annotation_position="top")
    fig.update_xaxes(title="Share of shots (%, log scale)", type="log", range=[-1.3, 2.05])
    fig.update_layout(barmode="group", hovermode="closest")
    return style_fig(fig, height)


def depth_fig(sweep, random_p, height=300):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[s["p"] for s in sweep], y=[100 * s["p_optimal"] for s in sweep],
                             mode="lines+markers", name="P(optimal) per shot", line=dict(color=TEAL, width=3),
                             marker=dict(size=9)), secondary_y=False)
    fig.add_trace(go.Bar(x=[s["p"] for s in sweep], y=[s["cx_gates"] for s in sweep], name="CX gates",
                         marker_color="rgba(140,150,163,.35)"), secondary_y=True)
    fig.add_hline(y=100 * random_p, line=dict(color=RED, dash="dot"), annotation_text="Random valid guess")
    fig.update_xaxes(title="QAOA depth p", dtick=1)
    fig.update_yaxes(title="P(optimal) %", secondary_y=False)
    fig.update_yaxes(title="CX gates", secondary_y=True, showgrid=False)
    return style_fig(fig, height)


def speedup_fig(speedups, height=280):
    names = [s["method"] for s in speedups]
    fig = go.Figure(go.Bar(y=names, x=[s["evaluations"] for s in speedups], orientation="h",
                           marker_color=[GREY, VIOLET, TEAL][: len(names)],
                           text=[f"{s['evaluations']} evaluations, P(opt) {100 * s['p_optimal']:.1f}%"
                                 f"{', optimum found' if s['found'] else ''}" for s in speedups],
                           textposition="auto"))
    fig.update_xaxes(title="Circuit evaluations to train (lower is faster)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return style_fig(fig, height, legend=False)


# ------------------------------------------------------------------ hardware

SOURCE_COLORS = {"Ideal simulator": TEAL, "Noisy simulator": "#7FC8BF", "IBM hardware": INK, "Random guess": BASE}


def hardware_fig(rows, metric="p_valid", height=360):
    """rows: list of dicts with experiment, source, p_valid, p_optimal."""
    exps = list(dict.fromkeys(r["experiment"] for r in rows))
    sources = list(dict.fromkeys(r["source"] for r in rows))
    fig = go.Figure()
    for s in sources:
        vals = []
        for e in exps:
            v = next((r[metric] for r in rows if r["experiment"] == e and r["source"] == s), None)
            vals.append(None if v is None else 100 * v)
        fig.add_trace(go.Bar(x=exps, y=vals, name=s, marker_color=SOURCE_COLORS.get(s.split(" (")[0], VIOLET),
                             hovertemplate="%{y:.1f}%"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=("Valid schedules" if metric == "p_valid" else "Optimal schedule") + " (% of shots)",
                     range=[0, 100])
    return style_fig(fig, height)


def counts_fig(top, height=300):
    colors = [TEAL if c["optimal"] else ("#7FC8BF" if c["valid"] else BASE) for c in top]
    total = sum(c["count"] for c in top)
    fig = go.Figure(go.Bar(x=[c["bits"] for c in top], y=[c["count"] for c in top], marker_color=colors,
                           customdata=[[c["cost"], "optimal" if c["optimal"] else ("valid" if c["valid"] else "breaks a rule")]
                                       for c in top],
                           hovertemplate="%{x}: %{y} shots<br>cost ₹%{customdata[0]}, %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title="Measured bitstring", type="category")
    fig.update_yaxes(title="Shots")
    fig.update_layout(hovermode="closest")
    return style_fig(fig, height, legend=False)
