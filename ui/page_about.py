import streamlit as st

from .style import load_json


def render():
    city = load_json("city_day.json") or {}
    meta = city.get("meta", {})
    st.title("About this project")
    st.markdown("""
**Quantum EV Charging Optimizer** is our entry for QHack 2026. It schedules electric-vehicle charging across a city
so that cars are charged on time while the local grid avoids new peaks.

### What is real and what is illustrative
- **Real:** the quantum circuits (Qiskit), the QAOA training, the forecasting model, the optimisation, the IBM
  device noise models and the IBM hardware runs. Every QAOA answer is checked against a brute-force solution.
- **Illustrative:** hub sizes, the fleet of vehicles and the feeder loads are simulated, built around real
  Bengaluru neighbourhoods. Swapping in real charging-session data (for example ACN-Data) needs no code changes
  in the optimiser.

### Honest limitations
- Each sub-problem has at most 16 qubits. At that size a classical solver is faster, so we do not claim quantum
  advantage. The contribution is a complete hybrid pipeline that already runs on quantum hardware and is
  structured to grow with it.
- Every charging session lasts one hour. Partial and multi-hour sessions are on the roadmap.
- Real hardware runs use small 4 and 6 qubit circuits, because deeper circuits are still dominated by noise.

### Roadmap
1. Multi-hour sessions and vehicle-to-grid discharge at peaks.
2. Larger sub-problems on hardware with error mitigation, and comparison with quantum annealing on the same QUBO.
3. Real session data and live feeder measurements from a partner charge-point operator.
4. A driver app with booking, reminders and the grid-friendly credit.
""")
    st.markdown("### Data and assumptions")
    st.markdown(f"""
- Time-of-day tariff: solar hours (09:00 to 16:00) at 0.7× and other hours at 1.3× an illustrative ₹7/kWh base,
  following the structure in India's EV charging guidelines as adopted by KSERC in 2025.
- CO₂: {meta.get('emission_kg_per_kwh', 0.727)} kg per kWh, the CEA CO₂ Baseline Database weighted average
  for FY 2023-24.
- Grid stress is priced at ₹{meta.get('gamma', 0.02)} per kW² per hour on each hub's feeder.
""")
    st.markdown("### References")
    st.markdown("""
- Farhi, Goldstone, Gutmann. *A Quantum Approximate Optimization Algorithm*, 2014.
- Hadfield et al. *From QAOA to a Quantum Alternating Operator Ansatz*, Algorithms, 2019.
- Barkoutsos et al. *Improving Variational Quantum Optimization using CVaR*, Quantum, 2020.
- Zhou et al. *QAOA: Performance, Mechanism, and Implementation on Near-Term Devices*, PRX, 2020.
- Egger, Mareček, Woerner. *Warm-starting quantum optimization*, Quantum, 2021.
- Galda et al. *Transferability of optimal QAOA parameters between random graphs*, IEEE QCE, 2021.
- Lucas. *Ising formulations of many NP problems*, Frontiers in Physics, 2014.
""")
    st.markdown("### Team")
    st.markdown("Minal Venkatesha Poppur, CMR Institute of Technology, and team.")
    st.markdown("[Source code on GitHub](https://github.com/MinalVP0824/EV-Quantum-Charging)")
