"""Quantum EV Charging Optimizer: Streamlit app entry point.

    streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")

import streamlit as st  # noqa: E402

from ui import nav, page_about, page_book, page_city, page_hardware, page_lab, page_overview  # noqa: E402
from ui.style import inject_css  # noqa: E402

st.set_page_config(page_title="Quantum EV Charging Optimizer", page_icon="⚡", layout="wide")
inject_css()

nav.PAGES.update({
    "overview": st.Page(page_overview.render, title="Overview", icon=":material/bolt:", url_path="overview", default=True),
    "city": st.Page(page_city.render, title="City day", icon=":material/map:", url_path="city"),
    "book": st.Page(page_book.render, title="Book a charge", icon=":material/ev_station:", url_path="book"),
    "lab": st.Page(page_lab.render, title="Quantum Lab", icon=":material/science:", url_path="lab"),
    "hardware": st.Page(page_hardware.render, title="IBM hardware", icon=":material/memory:", url_path="hardware"),
    "about": st.Page(page_about.render, title="About", icon=":material/info:", url_path="about"),
})

st.navigation(list(nav.PAGES.values()), position="top").run()
