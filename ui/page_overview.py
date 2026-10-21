import streamlit as st

from . import charts
from .nav import PAGES
from .style import load_json, show


def render():
    city = load_json("city_day.json")
    lab = load_json("lab_demo.json")
    if city is None:
        st.error("Precomputed results are missing. Run `python scripts/precompute.py` once, then reload.")
        return
    S = city["strategies"]
    fc, sm, qu = S["fcfs"]["kpis"], S["smart"]["kpis"], S["quantum"]["kpis"]
    qa = S["quantum"]["qaoa"]

    st.markdown(
        f"""<div class="hero-title">EVs push Bengaluru's evening peak up by
<span class="num-bad">{fc['peak_rise_kw']:.0f} kW</span>. Quantum-coordinated charging holds it to
<span class="num-good">{qu['peak_rise_kw']:.0f} kW</span>.</div>
<div class="lede">An AI model forecasts each feeder's load, and a quantum algorithm (QAOA) decides which
car charges where and when. Below is one simulated day with {city['meta']['n_vehicles']} EVs across six city hubs.</div>""",
        unsafe_allow_html=True,
    )
    show(charts.city_load_fig(city, height=430), key="hero")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Added to the city peak", f"{qu['peak_rise_kw']:.0f} kW",
              f"{qu['peak_rise_kw'] - fc['peak_rise_kw']:+.0f} kW vs charging on arrival", delta_color="inverse",
              border=True)
    c2.metric("Transformer overload hours", f"{qu['overload_hub_hours']}",
              f"{qu['overload_hub_hours'] - fc['overload_hub_hours']:+d} vs charging on arrival",
              delta_color="inverse", border=True)
    c3.metric("Charged in solar hours", f"{qu['solar_share_pct']:.0f}%",
              f"{qu['solar_share_pct'] - fc['solar_share_pct']:+.0f} pts", border=True)
    c4.metric("Matched the exact optimum", f"{qa['found_optimum_count']} of {qa['subproblems']}",
              f"worst gap {qa['max_gap_pct']}%", delta_color="off", border=True)

    st.markdown(
        f'<p class="note">Uncoordinated smart-charging apps, where every driver chases the cheapest slot, '
        f'still add {sm["peak_rise_kw"]:.0f} kW because they all pick the same hours. Coordination is what removes the peak. '
        f'Hub sizes, fleet and feeder loads are illustrative; tariffs follow India\'s solar/non-solar '
        f'time-of-day structure for EV charging.</p>',
        unsafe_allow_html=True,
    )

    st.subheader("How it works")
    steps = [
        ("Forecast", "A gradient-boosting model predicts every feeder's load hour by hour, with a P10 to P90 band, "
                     f"and corrects itself as real measurements arrive. Held-out error: "
                     f"{lab['forecast']['metrics']['mae_model_kw']:.2f} kW." if lab else "Gradient boosting with a P10 to P90 band."),
        ("Decompose", f"Each hour, cars that compete for the same chargers are grouped into small sub-problems of at "
                      f"most {qa['max_qubits']} qubits, so today's quantum hardware can handle a city."),
        ("Solve with QAOA", "A constraint-preserving XY-QAOA circuit searches only valid schedules. "
                            f"Angles carry over between sub-problems, cutting training from about "
                            f"{qa['avg_evals_cold_same_problems']:.0f} to {qa['avg_evals_transfer_same_problems']:.0f} circuit runs."),
        ("Book", "The best measured schedule becomes real bookings: which hub, which hour. "
                 "Drivers see their slot and what they save."),
    ]
    cols = st.columns(4)
    for i, (col, (title, text)) in enumerate(zip(cols, steps), start=1):
        col.markdown(f'<div class="step-n">{i}</div><div class="step-t">{title}</div>'
                     f'<div class="step-d">{text}</div>', unsafe_allow_html=True)

    st.divider()
    a, b, c, d = st.columns(4)
    a.page_link(PAGES["city"], label="Watch the day unfold", icon=":material/map:")
    b.page_link(PAGES["book"], label="Book a charge", icon=":material/ev_station:")
    c.page_link(PAGES["lab"], label="Open the Quantum Lab", icon=":material/science:")
    d.page_link(PAGES["hardware"], label="See the IBM hardware run", icon=":material/memory:")
