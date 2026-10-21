import pandas as pd
import streamlit as st

from . import charts
from .style import load_json, rs, show

VEHICLES = {
    "Compact hatchback (25 kWh battery)": 25,
    "Sedan (40 kWh)": 40,
    "SUV (60 kWh)": 60,
    "Fleet cab (30 kWh)": 30,
    "Delivery van (50 kWh)": 50,
}


@st.cache_data(show_spinner=False, max_entries=64)
def _book(energy, arrival, departure, pref):
    from evq.city import book_driver
    city = load_json("city_day.json")
    return book_driver(city, energy, arrival, departure, pref)


def render():
    city = load_json("city_day.json")
    if city is None:
        st.error("Precomputed results are missing. Run `python scripts/precompute.py` once, then reload.")
        return
    hubs = [h["name"] for h in city["hubs"]]

    st.title("Book a charge")
    st.markdown('<div class="lede">Tell us about your car and your day. The planner fits you in among the other '
                "drivers arriving in the same hour and solves that small problem with QAOA.</div>",
                unsafe_allow_html=True)

    form, result = st.columns([1, 1.5], gap="large")
    with form:
        with st.form("booking_form"):
            car = st.selectbox("Your vehicle", list(VEHICLES))
            now, target = st.slider("Battery now and target (%)", 0, 100, (25, 80), 5)
            where = st.selectbox("You are near", hubs, index=0)
            c1, c2 = st.columns(2)
            arrive = c1.selectbox("Arriving at", list(range(7, 23)), index=11, format_func=lambda h: f"{h:02d}:00")
            leave = c2.selectbox("Need it by", list(range(8, 24)), index=15, format_func=lambda h: f"{h:02d}:00")
            go_ = st.form_submit_button("Find my slot", type="primary", width="stretch")
        battery = VEHICLES[car]
        energy = round(battery * max(target - now, 0) / 100)
        st.caption(f"That is about {energy} kWh. Sessions are one hour; hubs deliver up to 60 kWh in that time.")

    if go_:
        if leave < arrive:
            st.session_state["booking_error"] = "Your leaving time is before your arrival. Pick a later time."
            st.session_state.pop("booking_result", None)
        elif energy <= 0:
            st.session_state["booking_error"] = "Your target charge must be above your current charge."
            st.session_state.pop("booking_result", None)
        else:
            with result:
                with st.spinner("Solving your sub-problem with QAOA..."):
                    st.session_state["booking_result"] = (_book(min(energy, 60), arrive, leave, hubs.index(where)),
                                                   dict(energy=min(energy, 60), arrive=arrive, leave=leave, where=where))
                st.session_state.pop("booking_error", None)

    with result:
        if st.session_state.get("booking_error"):
            st.warning(st.session_state["booking_error"])
        booked = st.session_state.get("booking_result")
        if not booked:
            st.info("Fill in the form and press **Find my slot**.")
            show(charts.city_map_fig(city, "quantum", height=360, animate=False, hour=18), key="book_map_empty",
                 modebar=False)
            return
        r, req = booked
        if not r["ok"]:
            st.warning(r["reason"] + " Try a later leaving time or another nearby hub.")
            return
        plan, naive, sav = r["plan"], r["naive"], r["savings"]
        st.markdown(f'<div class="plan-hub">Charge at {plan["hub"]}, {plan["hour"]:02d}:00</div>',
                    unsafe_allow_html=True)
        chips = []
        if plan["solar"]:
            chips.append("Solar hours tariff")
        if plan["detour_km"] > 0:
            chips.append(f"{plan['detour_km']:.0f} km round trip to this hub")
        if plan["wait_h"] == 0:
            chips.append("Charge right away")
        st.markdown(" ".join(f'<span class="chip">{c}</span>' for c in chips) or "&nbsp;", unsafe_allow_html=True)

        a, b, c = st.columns(3)
        diff = plan["energy_cost_rs"] - naive["energy_cost_rs"] if naive else 0
        a.metric("You pay", rs(plan["energy_cost_rs"]),
                 f"{'−' if diff < 0 else '+'}{rs(abs(diff))} vs charging now" if diff else None,
                 delta_color="inverse", border=True,
                 help=None if diff else "Same price as charging right away; the gain is on the grid side.")
        if sav:
            b.metric("Grid-friendly credit", rs(sav["credit_rs"]),
                     help="Proposed incentive: half of the grid cost you help avoid is credited to you.", border=True)
        c.metric("Ready by", f"{plan['hour'] + 1:02d}:00", border=True,
                 help=f"You said you leave at {req['leave']:02d}:00.")
        if naive:
            st.markdown(
                f"Charging immediately at **{naive['hub']}** would put your car on a feeder carrying "
                f"**{naive['feeder_load_kw']:.0f} kW**. Your slot's feeder carries **{plan['feeder_load_kw']:.0f} kW**, "
                f"which avoids about **{rs(max(sav['grid_relief_rs'], 0))}** of grid stress.")
        show(charts.city_map_fig(city, "quantum", height=320, animate=False, hour=plan["hour"],
                                 highlight=plan["hub_index"], pref=hubs.index(req["where"])), key="book_map",
             modebar=False)

        st.markdown("**Other slots we considered**")
        alt = pd.DataFrame([{"Hub": x["hub"], "Time": f"{x['hour']:02d}:00", "Price (₹/kWh)": x["tariff"],
                             "You pay (₹)": x["energy_cost_rs"], "Wait (h)": x["wait_h"],
                             "Feeder load (kW)": x["feeder_load_kw"], "Overall score (₹)": x["score_rs"]}
                            for x in r["alternatives"]])
        st.dataframe(alt, hide_index=True, width="stretch")
        qi = r["quantum"]
        with st.expander("How the quantum planner decided"):
            others = ", ".join(r["others_in_subproblem"]) or "nobody else"
            st.write(f"You shared a {qi['qubits']}-qubit sub-problem with {others}, the drivers arriving at "
                     f"{req['arrive']:02d}:00 who compete for the same chargers. XY-QAOA (depth 2, angles transferred "
                     f"from the city run) put {qi['p_optimal_pct']:.1f}% of its shots on the jointly optimal plan"
                     f"{' and matched the brute-force optimum' if qi['found_optimum'] else ''}. "
                     f"Training took {qi['evaluations']} circuit evaluations.")
            st.caption(f"Your session uses about {r['co2_kg']:.0f} kg of CO₂ at India's average grid intensity "
                       "(CEA, FY 2023-24). Charging in solar hours lowers the real figure.")
