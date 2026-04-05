# IRP-PD Heuristic Project

This repository contains the implementation developed for our graduation project on the Inventory Routing Problem with Pickup and Delivery (IRP-PD).

## Files
- `data_generation.py`: Generates multi-period demand data and creates Excel input files.
- `main_mtz.py`: MTZ-based MILP formulation.
- `main_lazy.py`: Lazy-constraint-based MILP formulation.
- `heuristic_main.py`: Improvement heuristic that improves incumbent MILP solutions.

## Notes
The project is based on the IRP-PD formulation inspired by Archetti et al.
