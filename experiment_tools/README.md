# Experiment Tools

This directory contains standalone scripts for orchestrating time-windowed
load experiments without modifying the main client/server runtime.

Main entrypoint:

- `python -m dnn_partition.experiment_tools.run_experiment --config ... --name ...`

Outputs per run:

- `experiment_plan.csv`
- `experiment_events.csv`
- `experiment_joined.csv`
