import pandas as pd
import streamlit as st

from . import charts
from .style import STRATEGY, load_json, show


@st.cache_resource(show_spinner=False, max_entries=3, ttl=6 * 3600)
def _simulate(n, seed, p90):
    from evq.city import simulate_day
    return simulate_day(n_vehicles=n, fleet_seed=seed, use_p90=p90, cold_compare=0)


def _kpi_table(city):
    rows = [
        ("Added to the city peak (kW)", "peak_rise_kw", "{:,.0f}"),
        ("City peak (kW)", "city_peak_kw", "{:,.0f}"),
        ("Transformer overload hours", "overload_hub_hours", "{:d}"),
        ("Energy bill for drivers (₹)", "energy_cost_rs", "{:,.0f}"),
        ("Grid stress cost (₹)", "grid_stress_rs", "{:,.0f}"),
        ("Total system cost (₹)", "system_cost_rs", "{:,.0f}"),
        ("Charged in solar hours (%)", "solar_share_pct", "{:.1f}"),
        ("Average wait before charging (h)", "avg_wait_h", "{:.2f}"),
        ("Extra driving to another hub (km)", "detour_km", "{:,.1f}"),
        ("Vehicles that could not be served", "unserved", "{:d}"),
    ]
    order = ["fcfs", "smart", "quantum", "exact"]
    data = {STRATEGY[s]["name"]: [fmt.format(city["strategies"][s]["kpis"][k]) for _, k, fmt in rows]
            for s in order if s in city["strategies"]}
    return pd.DataFrame(data, index=[r[0] for r in rows])


def render():
    pre = load_json("city_day.json")
    if pre is None:
        st.error("Precomputed results are missing. Run `python scripts/precompute.py` once, then reload.")
        return
    city = st.session_state.get("city_live") if st.session_state.get("use_live") else pre
    city = city or pre
    m = city["meta"]

    st.title("A day of EV charging in Bengaluru")
    st.markdown(f'<div class="lede">{m["n_vehicles"]} electric vehicles arrive at six hubs between 07:00 and 22:00. '
                "Every hour the planner re-forecasts the feeders and books the cars that just arrived. "
                "Press play on the map to watch the day.</div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        strat = st.segmented_control("Show on the map", ["quantum", "fcfs", "smart"], default="quantum",
                                     format_func=lambda s: STRATEGY[s]["name"], key="map_strat") or "quantum"
        show(charts.city_map_fig(city, strat, height=540), key=f"map_{strat}", modebar=False)
    with right:
        st.markdown("**City feeder load through the day**")
        show(charts.city_load_fig(city, height=270, compact=True), key="cityload")
        st.markdown("**Where and when EVs charged**")
        show(charts.hub_heatmap_fig(city, height=250), key="heat")

    st.subheader("Same day, four ways to run it")
    st.dataframe(_kpi_table(city), width="stretch")
    q, f = city["strategies"]["quantum"]["kpis"], city["strategies"]["fcfs"]["kpis"]
    st.markdown(
        f"Compared with charging on arrival, the quantum-coordinated plan adds **{q['peak_rise_kw']:.0f} kW** to the "
        f"city peak instead of **{f['peak_rise_kw']:.0f} kW**, removes all transformer overloads and saves drivers "
        f"**₹{f['energy_cost_rs'] - q['energy_cost_rs']:,.0f}** on the day. The price is a longer wait: "
        f"{q['avg_wait_h']:.1f} h on average, always before each driver's departure time.")

    st.subheader("EV load by hour")
    show(charts.ev_load_fig(city, height=300), key="evload")

    qa = city["strategies"]["quantum"].get("qaoa")
    if qa:
        st.subheader("Inside the quantum engine")
        a, b, c, d = st.columns(4)
        a.metric("QAOA sub-problems", qa["subproblems"], border=True)
        b.metric("Largest sub-problem", f"{qa['max_qubits']} qubits", f"average {qa['avg_qubits']}",
                 delta_color="off", border=True)
        c.metric("Solved to the exact optimum", f"{qa['found_optimum_count']} of {qa['subproblems']}",
                 f"worst gap {qa['max_gap_pct']}%", delta_color="off", border=True)
        if qa.get("avg_evals_cold_same_problems"):
            d.metric("Training runs with parameter transfer", f"{qa['avg_evals_transfer_same_problems']:.0f}",
                     f"vs {qa['avg_evals_cold_same_problems']:.0f} from scratch", delta_color="off", border=True)
        else:
            d.metric("Average training runs", f"{qa['avg_evals_transfer']:.0f}", border=True)
        fig = charts.qaoa_engine_fig(city)
        if fig is not None:
            show(fig, key="engine")
            st.caption("Each bar is one sub-problem solved by XY-QAOA (depth 2). Teal: the sampled schedule equals the "
                       "brute-force optimum. Yellow: a near-optimal schedule was used.")

    with st.expander("Simulate your own day"):
        st.write("Change the fleet and rerun the full day. It takes about a minute for 120 vehicles.")
        c1, c2, c3 = st.columns(3)
        n = c1.slider("Number of EVs", 40, 160, m["n_vehicles"], 10)
        seed = c2.number_input("Fleet seed", 1, 999, m["fleet_seed"])
        p90 = c3.toggle("Plan for the P90 forecast", False, help="Risk-averse planning against a high-load day.")
        run_col, toggle_col = st.columns([1, 2])
        if run_col.button("Simulate this day", type="primary"):
            with st.status("Simulating the day...", expanded=False) as status:
                st.session_state["city_live"] = _simulate(int(n), int(seed), bool(p90))
                st.session_state["use_live"] = True
                status.update(label="Done", state="complete")
            st.rerun()
        if st.session_state.get("city_live"):
            st.session_state["use_live"] = toggle_col.toggle("Show my simulated day", value=st.session_state.get("use_live", False))
