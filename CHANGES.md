# Upgrade for QHack Round 2

Replace your repository with the contents of this folder (or copy every file listed below).

## New files
- `evq/city.py` - city-scale digital twin: Bengaluru hubs, fleet, hourly re-planning, decomposition into <=16-qubit sub-problems, four strategies, driver booking
- `evq/hardware.py` - run trained circuits on IBM hardware or a noisy simulator and score the shots
- `evq/serialize.py` - JSON export so the app opens instantly
- `hardware/run_ibm.py` - command-line script for real IBM quantum hardware
- `scripts/precompute.py` - regenerates every result the app shows
- `ui/` - new multipage interface: `style.py`, `charts.py`, `nav.py`, `page_overview.py`, `page_city.py`, `page_book.py`, `page_lab.py`, `page_hardware.py`, `page_about.py`
- `.streamlit/config.toml` - theme (colours, fonts, light and dark)
- `static/fonts/` - bundled IBM Plex fonts (no external font server)
- `.github/workflows/tests.yml` - GitHub Actions: tests and an app start-up check on every push
- `tests/test_features.py` - 6 new tests
- `results/precomputed/city_day.json`, `results/precomputed/lab_demo.json`, `results/hardware/noisy_FakeFez.json`
- `docs/*.png` - screenshots used in the README

## Changed files
- `app.py` - now only sets up navigation; all pages live in `ui/`
- `evq/qaoa.py` - warm-start XY-QAOA, parameter transfer, fast sampling backend
- `evq/problem.py` - adds the 6-qubit `trio` instance used on hardware
- `requirements.txt` - adds plotly, qiskit-ibm-runtime, qiskit-aer
- `README.md` - rewritten for the new features
- `notebooks/EV_Quantum_Charging_Demo.ipynb` - re-executed (same results)

## Unchanged
`evq/data.py`, `evq/forecast.py`, `evq/baselines.py`, `evq/pipeline.py`, `evq/viz.py`, `run_pipeline.py`, `tests/test_core.py`, `LICENSE`, `.gitignore` (one line added)
