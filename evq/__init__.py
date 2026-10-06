"""evq: hybrid quantum + AI scheduling of EV charging.

Modules
  data       synthetic charging-site load
  forecast   gradient-boosted demand forecaster with P10/P90 band
  problem    instance definition, QUBO construction, schedule metrics
  qaoa       QAOA on Qiskit (standard X mixer and constraint-preserving XY mixer)
  baselines  FCFS, brute force, simulated annealing
  pipeline   end-to-end run used by the notebook, CLI and dashboard
  viz        figures
"""

__version__ = "0.1.0"
