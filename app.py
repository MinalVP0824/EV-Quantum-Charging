"""Streamlit dashboard for the hybrid quantum + AI EV charging scheduler.

    streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from evq import viz
from evq.pipeline import WINDOW_HOURS, run
from evq.problem import build_qubo, make_instance
from evq.qaoa import qaoa_circuit, qubo_to_ising

st.set_page_config(page_title="Quantum EV Charging Optimizer", page_icon="⚡", layout="wide")

st.title("⚡ Quantum EV Charging Optimizer")
st.caption("AI demand forecast → QUBO / Ising Hamiltonian → constraint-preserving QAOA (Qiskit) "
           "→ charging schedule, benchmarked against first-come-first-served and the exact optimum.")

with st.sidebar:
    st.header("Settings")
    instance = st.selectbox("Problem size", ["demo", "toy", "large"],
                            format_func={"toy": "Toy: 2 vehicles, 4 qubits",
                                         "demo": "Demo: 4 vehicles, 2 stations, 15 qubits",
                                         "large": "Large: 5 vehicles, 19 qubits (slow)"}.get)
    p = st.slider("QAOA depth p", 1, 6, 5 if instance != "large" else 2)
    alpha = st.select_slider("CVaR alpha", [0.1, 0.25, 0.5, 1.0], value=0.25,
                             help="1.0 = standard expectation value. Smaller values focus on the best outcomes.")
    gamma = st.slider("Peak-shaving weight (Rs per kW²)", 0.0, 0.06, 0.02, 0.005)
    use_p90 = st.toggle("Plan for P90 demand (risk-averse)", value=False)
    compare = st.toggle("Also run standard X-mixer QAOA", value=instance != "large")
    shots = st.select_slider("Shots", [1024, 2048, 4096, 8192], value=4096)
    seed = st.number_input("Seed", value=7, step=1)
    go = st.button("Run optimizer", type="primary", width="stretch")


@st.cache_resource(show_spinner=False)
def cached_run(instance, p, alpha, gamma, use_p90, compare, shots, seed):
    return run(instance, p=p, alpha=alpha, shots=shots, seed=seed, use_p90=use_p90,
               gamma=gamma, compare_x_mixer=compare)


if "ran" not in st.session_state:
    st.session_state.ran = False
if go:
    st.session_state.ran = True

if not st.session_state.ran:
    st.info("Pick settings in the sidebar and press **Run optimizer**. The demo instance takes about "
            "15 to 40 seconds on a laptop.")
    st.stop()

with st.spinner("Forecasting demand, building the Hamiltonian and training QAOA..."):
    r = cached_run(instance, p, alpha, gamma, use_p90, compare, shots, int(seed))

inst, q = r.instance, r.qubo
fc_m = r.metrics["FCFS"]
qa_m = r.metrics["Quantum (XY-QAOA)"]
qres = r.qaoa["Quantum (XY-QAOA)"]

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak load", f"{qa_m['peak_load_kw']:.1f} kW",
          f"{qa_m['peak_load_kw'] - fc_m['peak_load_kw']:+.1f} kW vs FCFS", delta_color="inverse")
c2.metric("Energy cost", f"Rs {qa_m['energy_cost_rs']:.0f}",
          f"{qa_m['energy_cost_rs'] - fc_m['energy_cost_rs']:+.0f} vs FCFS", delta_color="inverse")
c3.metric("Gap to exact optimum", f"{qres.gap_pct:.2f}%")
c4.metric("P(optimal) per shot", f"{100 * qres.p_optimal:.1f}%",
          f"{100 * qres.p_feasible:.0f}% of shots valid", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["1. Forecast", "2. Problem & Hamiltonian", "3. Quantum solver", "4. Results"])

with tab1:
    st.pyplot(viz.forecast_plot(r.forecast, WINDOW_HOURS[instance]), width="stretch")
    m = r.forecast.metrics
    a, b, c = st.columns(3)
    a.metric("Model MAE (held-out week)", f"{m['mae_model_kw']:.2f} kW")
    b.metric("Seasonal-naive MAE", f"{m['mae_seasonal_naive_kw']:.2f} kW")
    c.metric("Improvement", f"{m['improvement_pct']:.1f}%")
    st.write("Forecast base load used in the Hamiltonian:",
             pd.DataFrame({"Slot": inst.slot_labels, "Base load (kW)": [round(x, 1) for x in inst.base_load],
                           "Tariff (Rs/kWh)": inst.tariff}))

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Vehicles")
        st.dataframe(pd.DataFrame([{
            "Vehicle": v.name, "Energy (kWh)": v.energy_kwh,
            "Arrives": inst.slot_labels[v.arrival], "Last slot": inst.slot_labels[v.departure],
            "Detour cost": ", ".join(f"{inst.stations[s].name}: Rs {c:.0f}"
                                     for s, c in v.detour.items()) or "none",
        } for v in inst.vehicles]), hide_index=True, width="stretch")
        st.subheader("Stations")
        st.dataframe(pd.DataFrame([{"Station": s.name, "Max kW": s.max_kw} for s in inst.stations]),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("QUBO → Ising")
        h, J, _ = qubo_to_ising(q.Q, q.offset)
        st.markdown(f"""
- **Qubits:** {q.n} (one per allowed vehicle, station, slot choice)
- **Ising terms:** {int((abs(h) > 1e-12).sum())} single-Z, {len(J)} ZZ couplings
- **Penalty weight:** {q.penalty:.1f}
- **Valid schedules:** {int(r.feasible.sum())} of {2 ** q.n:,} bitstrings
""")
        st.markdown("**Cost Hamiltonian** (Rs)")
        st.latex(r"H_{\text{cost}} = \sum_{v,s,t}\big(\tau_t E_v + w\,\Delta t_v + \delta_{v,s}\big)\,x_{vst}")
        st.latex(r"\;+\; \gamma\sum_t\Big(\hat d_t + \sum_{v,s}E_v\,x_{vst}\Big)^2")
        st.markdown("**Constraint penalties**")
        st.latex(r"H_A = A\sum_v\Big(1-\sum_{s,t}x_{vst}\Big)^2 \qquad "
                 r"H_B = B\sum_{s,t}\sum_{v<v'}x_{vst}\,x_{v'st}")
        st.caption("d̂ₜ is the AI forecast. It enters the Hamiltonian through the grid-peak term, so the "
                   "prediction directly shapes the energy landscape QAOA explores.")

with tab3:
    st.pyplot(viz.convergence_plot(r.qaoa), width="stretch")
    a, b = st.columns(2)
    with a:
        st.pyplot(viz.distribution_plot(qres, r.energies, r.feasible), width="stretch")
    with b:
        if len(r.qaoa) > 1:
            st.pyplot(viz.mixer_comparison_plot(r.quantum_table.assign(
                prob_optimal=r.quantum_table["prob_optimal"].replace(0, float("nan")))),
                width="stretch")
    st.dataframe(r.quantum_table, hide_index=True, width="stretch")
    with st.expander("Circuit structure (one XY-QAOA layer on the 4-qubit toy instance)"):
        toy = make_instance("toy", inst.base_load[:2])
        tq = build_qubo(toy)
        th, tJ, _ = qubo_to_ising(tq.Q_xy, tq.offset_obj)
        small, _ = qaoa_circuit(th, tJ, tq.n, 1, tq.groups)
        st.pyplot(small.draw("mpl", fold=-1, scale=0.8), width="stretch")
        st.caption("W-state preparation per vehicle → cost layer (RZ, RZZ) → XY mixer (XX+YY). "
                   "The mixer only swaps a vehicle's choice, so it can never break the 'charge exactly once' rule.")

with tab4:
    st.pyplot(viz.schedule_grid(inst, {k: r.assignments[k] for k in
                                       ["FCFS", "Classical optimum", "Quantum (XY-QAOA)"]}),
              width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(viz.load_plot(inst, {k: r.metrics[k] for k in ["FCFS", "Quantum (XY-QAOA)"]}),
                  width="stretch")
    with right:
        st.markdown(f"""
**What changed versus first-come-first-served**

- Peak feeder load: **{fc_m['peak_load_kw']:.1f} → {qa_m['peak_load_kw']:.1f} kW**
- Energy bill: **Rs {fc_m['energy_cost_rs']:.0f} → Rs {qa_m['energy_cost_rs']:.0f}**
- Average wait: {fc_m['avg_wait_slots']:.2f} → {qa_m['avg_wait_slots']:.2f} slots
- Every vehicle still charges before its deadline: **{'yes' if qa_m['feasible'] else 'no'}**

The optimizer moves flexible vehicles out of the forecast evening peak into the
cheaper, lightly loaded 20:00 slot, and steers drivers away from detours.
""")
    st.dataframe(r.table, hide_index=True, width="stretch")

plt.close("all")
