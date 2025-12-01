# Discrete AE-4 Optimizer

A DEAP-driven evolutionary optimizer for proposing synthetic tube networks that balance coverage and cost using Transport for London demand data.

## Repository structure

| File | Purpose |
| --- | --- |
| `optimizer.py` | Core entry point. Defines `Station`, `TubeLine`, toolbox setup, and the genetic operators (init, mate, mutate). The evaluate function computes commute distances from the current network via a custom D'Esopo–Pape search and scores it against per-station demand. CLI flags expose population size, mutation/crossover probabilities, temperature scaling, budget parameters, and map-output options. |
| `station_layout.py` | Provides schematic coordinates for ~100 London stations, exports `STATION_ORDER`/`STATION_DISTANCES`, and exposes `build_layout` for mapping station names to coordinates (with a fallback grid for unknowns). |
| `filtered_OD.csv` | Cleaned origin-destination matrix limited to Zone 1 trips. Used by default when `optimizer.py` loads demand. |
| `OD_matrix_2017.csv` | Raw TfL data; use `Zone1-Filter.py` or your own script to produce filtered CSVs in the same format. |
| `Zone1-Filter.py` | Simple filter script—change the constants at the top (`input_file`, `output_file`, `RENDER_STYLE`) to generate DOT representations or trimmed OD matrices centered on Zone 1. |
| `station_graph_w.dot` | DOT graph exported from the filtered demand data. Serve it as a baseline graph for manual inspection or seeding an initial population. |
| `markov.py` | (Optional) Markov-chain-based word generator included for experimenting with line naming or procedural text generation. |
| `requirements.txt` | Python dependencies required to run the optimizer and rendering stack (`numpy`, `matplotlib`, `deap`, etc.). |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python optimizer.py --generations 400 --pop-size 150 --temperature 1.8 --seed 1119280223 --map-output optimized_map_test.svg
```

### CLI highlights

- `--data`: Path to the OD matrix. Defaults to `filtered_OD.csv`. Provide another CSV with the same column layout to retarget demand.
- `--cost-per-connection` & `--max-cost`: Tune the budget shared by every `TubeLine` in an individual. Each connection's cost scales with the squared distance from `STATION_DISTANCESCPU`.
- `--mutpb` / `--cxpb`: Control how often mutation/crossover happen when evolving the population. Multiply by `--temperature` to bias exploration.
- `--map-output`: Where to save the generated transit map (set to `""` to skip rendering).

## Rendering DOT files

Install Graphviz (`brew install graphviz` on macOS) and render the supplied DOT graphs with:

```bash
dot -Tsvg station_map.dot -o station_map.svg
open station_map.svg
```

Repeat the command with `station_graph_w.dot` if you want to visualize the Zone 1 graph derived from the OD data.

## Tips

1. **Customize mutation intensity**: The mutate operator uses `indpb` (default 0.25) plus per-line heuristics—raise `mutpb`/`indpb` for more aggressive exploration.
2. **Seed realistic layouts**: Use `station_graph_w.dot` or `station_map.dot` to craft hand-assembled individuals before running the GA.
3. **Add tests**: Tools like `mypy`/`pytest` can validate `build_connectivity_tensor`, `build_commute_distance_tensor`, and demand loading once you iron out NumPy conversions.
4. **Profile costs**: Each edge cost equals `cost_per_connection * (distance ** 2)`; adjusting `COST_PER_CONNECTION_DEFAULT` lets you favor shorter or longer links.
