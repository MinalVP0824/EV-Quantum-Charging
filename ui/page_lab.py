import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import charts
from .style import ROOT, TEAL, load_json, show, style_fig


def _live_run(p, alpha, compare, placeholder):
    from evq.pipeline import run
    from evq.serialize import pipeline_to_dict

    hist = []

    def cb(n, val):
        hist.append(val)
        if n % 20 == 0:
            fig = go.Figure(go.Scatter(y=hist, mode="lines", line=dict(color=TEAL, width=2)))
            fig.update_xaxes(title="Circuit evaluation")
            fig.update_yaxes(title="CVaR cost (₹)")
            placeholder.plotly_chart(style_fig(fig, 260, legend=False, title="Training XY-QAOA live"),
                                     width="stretch", key=f"live_{n}")

    r = run("demo", p=p, alpha=alpha, compare_x_mixer=compare, callback=cb)
    return pipeline_to_dict(r, "demo")


def render():
    pre = load_json("lab_demo.json")
    if pre is None:
        st.error("Precomputed results are missing. Run `python scripts/precompute.py` once, then reload.")
        return
    lab = st.session_state.get("lab_live") or pre

    st.title("Quantum Lab")
    st.markdown('<div class="lede">One sub-problem under the microscope: 4 vehicles, 2 hubs, 3 evening slots, '
                "15 qubits. Follow it from forecast to Hamiltonian to the trained circuit.</div>",
                unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1])
        p = c1.slider("QAOA depth p", 1, 6, 5)
        alpha = c2.select_slider("CVaR alpha", [0.1, 0.25, 0.5, 1.0], value=0.25,
                                 help="1.0 is the standard expectation value; smaller values train on the best outcomes.")
        compare = c3.toggle("Also train standard X-mixer QAOA", value=True)
        c4.write("")
        train = c4.button("Train live", type="primary", width="stretch")
        live_slot = st.empty()
        if train:
            with st.spinner("Training on the simulator, about 15 to 60 seconds..."):
                st.session_state["lab_live"] = _live_run(p, alpha, compare, live_slot)
            st.rerun()
        if st.session_state.get("lab_live"):
            if st.button("Back to the saved run"):
                st.session_state.pop("lab_live")
                st.rerun()

    xy = lab["qaoa"]["Quantum (XY-QAOA)"]
    s = xy["summary"]
    fcfs, qm = lab["metrics"]["FCFS"], lab["metrics"]["Quantum (XY-QAOA)"]
    a, b, c, d = st.columns(4)
    a.metric("Peak load", f"{qm['peak_load_kw']:.1f} kW", f"{qm['peak_load_kw'] - fcfs['peak_load_kw']:+.1f} kW vs FCFS",
             delta_color="inverse", border=True)
    b.metric("Energy cost", f"₹{qm['energy_cost_rs']:.0f}", f"{qm['energy_cost_rs'] - fcfs['energy_cost_rs']:+.0f} vs FCFS",
             delta_color="inverse", border=True)
    c.metric("Gap to exact optimum", f"{s['gap_pct']:.2f}%", border=True)
    d.metric("Shots on the optimal schedule", f"{100 * s['prob_optimal']:.1f}%",
             f"{100 * s['prob_feasible']:.0f}% of shots valid", delta_color="off", border=True)

    t1, t2, t3, t4, t5 = st.tabs(["Forecast", "Problem and Hamiltonian", "Quantum solver", "Speed-ups", "Schedules"])

    with t1:
        show(charts.forecast_fig(lab["forecast"]), key="lab_fc")
        m = lab["forecast"]["metrics"]
        x, y, z = st.columns(3)
        x.metric("Model error on a held-out week", f"{m['mae_model_kw']:.2f} kW", border=True)
        y.metric("Same hour yesterday (baseline)", f"{m['mae_seasonal_naive_kw']:.2f} kW", border=True)
        z.metric("Improvement", f"{m['improvement_pct']:.0f}%", border=True)

    with t2:
        left, right = st.columns(2)
        inst = lab["instance"]
        with left:
            st.markdown("**Vehicles**")
            st.dataframe(pd.DataFrame([{"Vehicle": v["name"], "Energy (kWh)": v["energy_kwh"], "Arrives": v["arrival"],
                                        "Last slot": v["last_slot"],
                                        "Detour": ", ".join(f"{k}: ₹{c:.0f}" for k, c in v["detour"].items()) or "none"}
                                       for v in inst["vehicles"]]), hide_index=True, width="stretch")
            st.markdown("**Slots**")
            st.dataframe(pd.DataFrame({"Slot": inst["slots"], "Forecast base load (kW)": inst["base_load"],
                                       "Tariff (₹/kWh)": inst["tariff"]}), hide_index=True, width="stretch")
        with right:
            qb = lab["qubo"]
            st.markdown(f"**{qb['qubits']} qubits**, one per allowed (vehicle, hub, slot) choice. "
                        f"{qb['z_terms']} single-qubit and {qb['zz_terms']} two-qubit Ising terms. "
                        f"Only {qb['valid_schedules']} of {qb['bitstrings']:,} bitstrings are valid schedules.")
            st.latex(r"H_{\text{cost}} = \sum_{v,s,t}\big(\tau_t E_v + w\,\Delta t_v + \delta_{v,s}\big)\,x_{vst}"
                     r" + \gamma\sum_t\Big(\hat d_t + \sum_{v,s}E_v\,x_{vst}\Big)^2")
            st.latex(r"H_A = A\sum_v\Big(1-\sum_{s,t}x_{vst}\Big)^2 \qquad H_B = B\sum_{s,t}\sum_{v<v'}x_{vst}\,x_{v'st}")
            st.caption("d̂ₜ is the AI forecast. It enters through the grid-peak term, so the prediction reshapes the "
                       "energy landscape QAOA explores. The XY mixer enforces H_A by construction.")
            img = ROOT / "results" / "07_circuit_one_layer_toy.png"
            if img.exists():
                st.image(str(img), caption="One XY-QAOA layer on a 4-qubit instance: W-state preparation, "
                                           "cost layer (RZ, RZZ), XY mixer (XX+YY).")

    with t3:
        names = list(lab["qaoa"])
        cols = st.columns(len(names))
        for col, n in zip(cols, names):
            with col:
                q = lab["qaoa"][n]
                show(charts.convergence_fig(q["history"], q["optimal_energy"], height=270,
                                            color=TEAL if "XY" in n else "#6C5DD3", title=f"{n}, p={q['summary']['p']}"),
                     key=f"conv_{n}")
        l, r = st.columns(2)
        with l:
            show(charts.distribution_fig(xy["distribution"], title="What the XY-QAOA circuit measures, by schedule cost"),
                 key="dist")
        with r:
            show(charts.mixer_fig(lab["qaoa"], pre["random_valid_p_optimal"]), key="mixer")
            st.caption("The constraint-preserving XY mixer never leaves the space of 'each car charges exactly once', "
                       "so far more shots land on valid and optimal schedules than with the standard mixer.")
        if "depth_sweep" in pre:
            show(charts.depth_fig(pre["depth_sweep"], pre["random_valid_p_optimal"]), key="depth")
        st.dataframe(pd.DataFrame([{"Solver": n, **{k: v for k, v in lab["qaoa"][n]["summary"].items() if k != "params"}}
                                   for n in names]), hide_index=True, width="stretch")

    with t4:
        if "speedups" in pre:
            show(charts.speedup_fig(pre["speedups"]), key="speed")
            st.markdown(
                "- **Warm start** biases each vehicle's qubits towards its individually cheapest option instead of an "
                "even superposition, so the circuit starts near a good schedule.\n"
                "- **Parameter transfer** reuses QAOA angles trained on earlier sub-problems. Normalised Hamiltonians "
                "from the same city look alike, so good angles carry over and only a short fine-tune is needed. "
                "This is what makes solving a whole day of sub-problems practical.\n\n"
                "Warm start raises the share of optimal shots; transfer cuts training about tenfold for a small drop "
                "in that share. In every case the best sampled schedule is still the exact optimum.")

    with t5:
        rows = []
        for method, a_ in lab["assignments"].items():
            for veh, (hub, slot) in a_.items():
                rows.append({"Method": method, "Vehicle": veh, "Hub": hub, "Slot": slot})
        grid = pd.DataFrame(rows).pivot(index="Vehicle", columns="Method", values="Slot")
        hubs = pd.DataFrame(rows).pivot(index="Vehicle", columns="Method", values="Hub")
        show(charts.slot_load_fig(lab), key="slotload")
        st.markdown("**Slot per vehicle**")
        st.dataframe(grid, width="stretch")
        st.markdown("**Hub per vehicle**")
        st.dataframe(hubs, width="stretch")
        st.dataframe(pd.DataFrame(lab["table"]), hide_index=True, width="stretch")
