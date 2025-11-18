# Discrete AE-4 Optimizer

This repository now includes a DEAP-based evolutionary optimizer that designs synthetic tube connections using the filtered Origin–Destination matrix supplied by Transport for London data.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python optimizer.py --generations 400 --pop-size 150 --temperature 1.8 --seed 1119280223 --map-output optimized_map_test.svg
```

The defaults enforce a cost-per-connection of 10 and a maximum budget of 400, so every individual in the population can contain at most 40 station-to-station links at any time. After each run the best individual is summarized and a stylized tube map is exported to `optimized_map.svg` (white background, horizontal A4 layout).

## How it works

- `filtered_OD.csv` is parsed into `Station` objects and aggregated, undirected `DemandPair` records (A→B is treated the same as B→A).
- `TubeLine` objects store ordered lists of stations. Each consecutive pair represents one connection with an associated cost.
- Individuals are lists of `TubeLine` objects. Custom `individual`, `mate`, and `mutate` operators make sure the connection budget is never exceeded.
- The `evaluate` function builds a `StationGraph`, runs a breadth-first search between every demand pair, multiplies the station-count distance by the total directional demand, and sums the results. Missing paths incur a large penalty, so lower scores mean better coverage.

## Tuning knobs

- `--cost-per-connection` and `--max-cost` control the budget shared by all lines in each individual.
- `--cxpb`, `--mutpb`, `--pop-size`, `--temperature`, and `--generations` expose standard DEAP evolutionary parameters.
- You can point `--data` to a different OD matrix (matching the existing column order) to retarget the experiment.
- `--map-output` chooses where to save the matplotlib-rendered transit map (set to `""` to skip rendering).

## Next steps

- Inject a real-world base graph (for example by parsing `station_graph_w.dot`) to seed the population with realistic adjacencies.
- Persist the best individual as a DOT/GeoJSON artifact for alternative visualizations.
- Add automated tests that validate both the BFS scoring and the rendering pipeline on toy graphs.
