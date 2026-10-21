import pandas as pd
import streamlit as st

from . import charts
from .style import show

LABELS = {("toy", "XY"): "4 qubits, XY mixer", ("toy", "X"): "4 qubits, X mixer",
          ("trio", "XY"): "6 qubits, XY mixer", ("trio", "X"): "6 qubits, X mixer"}


@st.cache_data(show_spinner=False, ttl=600)
def _load():
    from evq.hardware import load_all
    return load_all()


def render():
    st.title("Running on a real quantum computer")
    st.markdown('<div class="lede">The trained circuits are sent to IBM quantum processors. Noise is the big '
                "question for any quantum algorithm today, so we compare the ideal simulator, IBM's device noise model "
                "and the real hardware on the same circuits.</div>", unsafe_allow_html=True)

    runs = [r for r in _load() if not r.get("pending")]
    if not runs:
        st.info("No hardware results yet. Run `python hardware/run_ibm.py --fake FakeFez` for the noisy simulator "
                "or follow the steps below for real hardware.")
    hw = [r for r in runs if r["meta"]["kind"] == "IBM quantum hardware"]
    if runs and not hw:
        st.warning("Real-hardware results are not in yet. The charts show the ideal simulator and IBM's Heron noise "
                   "model. Run the steps at the bottom of this page to add a real device.")

    rows, details = [], []
    if runs:
        for e in runs[0]["experiments"]:
            name = LABELS.get((e["name"], e["mixer"]), f"{e['name']} {e['mixer']}")
            rows.append({"experiment": name, "source": "Ideal simulator", "p_valid": e["ideal_p_valid"],
                         "p_optimal": e["ideal_p_optimal"]})
        for r in sorted(runs, key=lambda r: r["meta"]["kind"] != "noisy simulator"):
            src = "IBM hardware" if r["meta"]["kind"] == "IBM quantum hardware" else "Noisy simulator"
            src = f"{src} ({r['meta']['backend']})"
            for e in r["experiments"]:
                name = LABELS.get((e["name"], e["mixer"]), f"{e['name']} {e['mixer']}")
                rows.append({"experiment": name, "source": src, "p_valid": e["p_valid"], "p_optimal": e["p_optimal"]})
                details.append({"Source": src, "Experiment": name, "Qubits": e["qubits"],
                                "Two-qubit gates": e["two_qubit_gates"], "Depth": e["transpiled_depth"],
                                "Valid %": round(100 * e["p_valid"], 1), "Optimal %": round(100 * e["p_optimal"], 1),
                                "Found optimum": "yes" if e["found_optimum"] else "no",
                                "Distance from ideal (TVD)": e["tvd_vs_ideal"],
                                "Job": (r["meta"].get("job_id") or "local") if "hardware" in src.lower() else "local simulation",
                                "_top": e["top_counts"]})
        for e in runs[0]["experiments"]:
            name = LABELS.get((e["name"], e["mixer"]), f"{e['name']} {e['mixer']}")
            rows.append({"experiment": name, "source": "Random guess", "p_valid": e["random_p_valid"],
                         "p_optimal": e["random_p_optimal"]})

        l, r = st.columns(2)
        with l:
            st.markdown("**Shots that are valid schedules**")
            show(charts.hardware_fig(rows, "p_valid", height=360), key="hw_valid")
        with r:
            st.markdown("**Shots on the optimal schedule**")
            show(charts.hardware_fig(rows, "p_optimal", height=360), key="hw_opt")

        six = [d for d in details if d["Experiment"] == "6 qubits, XY mixer"]
        six_x = [d for d in details if d["Experiment"] == "6 qubits, X mixer"]
        if six and six_x:
            all_found = all(d["Found optimum"] == "yes" for d in details)
            st.markdown(f"On the 6-qubit problem ({six[-1]['Source']}), the XY mixer keeps **{six[-1]['Valid %']:.0f}%** "
                        f"of shots valid against **{six_x[-1]['Valid %']:.0f}%** for standard QAOA"
                        + (", and every run still finds the optimal schedule among its samples." if all_found else "."))

        st.dataframe(pd.DataFrame([{k: v for k, v in d.items() if k != "_top"} for d in details]),
                     hide_index=True, width="stretch")

        pick = st.selectbox("Look at the raw measurements", [f"{d['Source']}: {d['Experiment']}" for d in details])
        chosen = next(d for d in details if f"{d['Source']}: {d['Experiment']}" == pick)
        show(charts.counts_fig(chosen["_top"]), key="hw_counts")
        st.caption("Teal: optimal schedule. Light teal: valid schedule. Grey: breaks a rule.")

    with st.expander("Run it on IBM hardware yourself", expanded=not hw):
        st.markdown("""
1. Create a free account at [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) and an instance on the **Open Plan**.
2. Copy your **API key** and the instance **CRN**.
3. In the project folder, with your virtual environment active:
""")
        st.code("python hardware/run_ibm.py --save-account --token YOUR_API_KEY --instance YOUR_CRN\n"
                "python hardware/run_ibm.py", language="bash")
        st.markdown("""
4. Commit the new file in `results/hardware/` and push. This page picks it up automatically.

The job runs 4 small circuits with 4,096 shots each and uses a few seconds of the free monthly QPU time.
If the queue is long, use `--submit-only` and collect later with `--retrieve JOB_ID`.
""")
