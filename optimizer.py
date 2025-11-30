"""Optimization framework for designing tube connections with DEAP.

This module wires together:

* Domain data structures for stations, tube lines, and passenger demand pairs
* Helper utilities for generating, mutating, and mating individuals under
  a strict cost-per-connection budget
* An evaluation function that scores networks via tensorized demand-distance products

Running this file directly will execute a small evolutionary search using the
`filtered_OD.csv` matrix bundled with the repository.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.path import Path as MplPath
from deap import algorithms, base, creator, tools
import torch
import numpy as np
from collections import deque



from station_layout import build_layout, STATION_DISTANCES, STATION_ORDER, DEVICE

STATION_INDEX = {name: idx for idx, name in enumerate(STATION_ORDER)}
STATION_DISTANCES_CPU = STATION_DISTANCES.detach().cpu()

COST_PER_CONNECTION_DEFAULT = 3
MAX_COST_DEFAULT = 10000000
MAX_LINE_LENGTH = 7  # maximum stations allowed within a single synthetic line
MAX_NUMBER_LINES = 11  # maximum number of lines allowed per individual
DISCONNECT_MULTIPLIER = 10000  # penalty multiplier when no route exists
DISTANCE_SCALAR_EPS = 1e-4

# Rendering constants
LINE_WIDTH_PT = 5.0
PARALLEL_OFFSET = 5  # separation between parallel lines on shared segments
CORNER_RADIUS = 1.0  # rounded corner radius for orthogonal polylines
STATION_BUFFER = 1  # distance to travel straight before/after stations


@dataclass(frozen=True)
class Station:
	"""Represents a named station parsed from the OD matrix."""

	code: int
	name: str


@dataclass
class TubeLine:
	"""A synthetic tube line consisting of an ordered list of stations."""

	name: str
	stations: List[str] = field(default_factory=list)

	def __post_init__(self) -> None:
		if len(self.stations) < 2:
			raise ValueError("TubeLine requires at least two stations")

	@property
	def connections(self) -> int:
		return max(0, len(self.stations) - 1)

	def edges(self) -> List[Tuple[str, str]]:
		return list(zip(self.stations[:-1], self.stations[1:]))

	def clone(self, suffix: str = "") -> "TubeLine":
		return TubeLine(name=f"{self.name}{suffix}", stations=list(self.stations))


def load_station_demands(csv_path: Path) -> Tuple[Dict[str, Station], torch.Tensor]:
	"""Parse the filtered OD matrix and build a symmetric OD demand tensor."""

	stations: Dict[str, Station] = {}
	demand_tensor = torch.zeros_like(STATION_DISTANCES, device=DEVICE)

	with csv_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		for row in reader:
			if len(row) < 11:
				continue
			try:
				origin_code = int(row[0])
				origin_name = row[1].strip()
				dest_code = int(row[2])
				dest_name = row[3].strip()
				total = int(row[-1])
			except ValueError:
				continue

			if origin_name == dest_name:
				continue

			origin_idx = STATION_INDEX.get(origin_name)
			dest_idx = STATION_INDEX.get(dest_name)
			if origin_idx is None or dest_idx is None:
				continue

			stations.setdefault(origin_name, Station(origin_code, origin_name))
			stations.setdefault(dest_name, Station(dest_code, dest_name))

			demand_tensor[origin_idx, dest_idx] += total
			demand_tensor[dest_idx, origin_idx] += total

	return stations, demand_tensor


line_counter = count(1)


def next_line_name(prefix: str = "L") -> str:
	return f"{prefix}{next(line_counter)}"


def total_connections(lines: Sequence[TubeLine]) -> int:
	return sum(line.connections for line in lines)


def connection_cost(a: str, b: str, cost_per_connection: float) -> float:
	idx_a = STATION_INDEX.get(a)
	idx_b = STATION_INDEX.get(b)
	if idx_a is None or idx_b is None:
		return 0.0
	distance = float(STATION_DISTANCES_CPU[idx_a, idx_b].item())
	return  cost_per_connection * (distance ** 2)


def total_connection_cost(lines: Sequence[TubeLine], cost_per_connection: float) -> float:
	total = 0.0
	for line in lines:
		for a, b in line.edges():
			total += connection_cost(a, b, cost_per_connection)
	return total


def remaining_budget(lines: Sequence[TubeLine], max_cost: float, cost_per_connection: float) -> float:
	return max(0.0, max_cost - total_connection_cost(lines, cost_per_connection))


def build_random_line(station_names: Sequence[str], max_segments: int) -> TubeLine:
	length = random.randint(2, max_segments)
	chosen = random.sample(station_names, length)
	return TubeLine(name=next_line_name(), stations=list(chosen))


def enforce_budget(lines: List[TubeLine], max_cost: float, cost_per_connection: float) -> None:
	while lines and total_connection_cost(lines, cost_per_connection) > max_cost:
		victim = random.choice(lines)
		if victim.connections <= 1:
			lines.remove(victim)
		else:
			victim.stations.pop()
			if len(victim.stations) < 2:
				lines.remove(victim)


def enforce_line_limit(lines: List[TubeLine], max_lines: int) -> None:
	while len(lines) > max_lines:
		lines.pop(random.randrange(len(lines)))


def mutate_line(line: TubeLine, station_names: Sequence[str]) -> None:
	action = random.random()
	if action < 0.4 and len(line.stations) > 2:
		idx = random.randrange(1, len(line.stations) - 1)
		del line.stations[idx]
	elif action < 0.7 and len(line.stations) < MAX_LINE_LENGTH:
		insert_pos = random.randrange(1, len(line.stations))
		candidate = random.choice(station_names)
		line.stations.insert(insert_pos, candidate)
	else:
		idx = random.randrange(len(line.stations))
		candidate = random.choice(station_names)
		line.stations[idx] = candidate


def init_individual(icls, station_names: Sequence[str], max_cost: float, cost_per_connection: float):
	lines: List[TubeLine] = []
	max_segments = max(2, min(MAX_LINE_LENGTH, len(station_names)))
	if max_segments < 2:
		raise RuntimeError("Not enough stations to seed an individual")

	while len(lines) < MAX_NUMBER_LINES:
		line = build_random_line(station_names, max_segments)
		lines.append(line)
		enforce_budget(lines, max_cost, cost_per_connection)
		if remaining_budget(lines, max_cost, cost_per_connection) <= 0 or random.random() < 0.25:
			break

	if not lines:
		line = build_random_line(station_names, 2)
		lines.append(line)
		enforce_budget(lines, max_cost, cost_per_connection)
		if not lines:
			raise RuntimeError("Unable to initialize an individual within the budget constraints")

	enforce_line_limit(lines, MAX_NUMBER_LINES)

	return icls(lines)


def mate_individual(ind1, ind2, max_cost: float, cost_per_connection: float):
	if not ind1 or not ind2:
		return ind1, ind2

	cx1 = random.randint(1, len(ind1))
	cx2 = random.randint(1, len(ind2))

	child1 = [line.clone("_A") for line in ind1[:cx1]] + [line.clone("_B") for line in ind2[cx2:]]
	child2 = [line.clone("_A") for line in ind2[:cx2]] + [line.clone("_B") for line in ind1[cx1:]]

	ind1[:] = child1
	ind2[:] = child2

	enforce_budget(ind1, max_cost, cost_per_connection)
	enforce_budget(ind2, max_cost, cost_per_connection)
	enforce_line_limit(ind1, MAX_NUMBER_LINES)
	enforce_line_limit(ind2, MAX_NUMBER_LINES)

	return ind1, ind2


def mutate_individual(
	individual,
	station_names: Sequence[str],
	max_cost: float,
	cost_per_connection: float,
	indpb: float = 0.2,
):
	for line in list(individual):
		if random.random() < indpb:
			mutate_line(line, station_names)
			if len(line.stations) < 2:
				individual.remove(line)

	if individual and random.random() < 0.2:
		individual.pop(random.randrange(len(individual)))

	remaining = remaining_budget(individual, max_cost, cost_per_connection)
	if len(individual) < MAX_NUMBER_LINES and remaining > 0 and random.random() < 0.6:
		max_segments = min(MAX_LINE_LENGTH, len(station_names))
		if max_segments >= 2:
			individual.append(build_random_line(station_names, max_segments))

	enforce_budget(individual, max_cost, cost_per_connection)
	enforce_line_limit(individual, MAX_NUMBER_LINES)
	return (individual,)

### Builds a copy of station distances where only connected lines are represented as fractions
### For n lines connecting a and b stations, the output tensor will have 1/n in (a,b) and (b,a)
def build_connectivity_tensor(individual: Sequence[TubeLine]) -> torch.Tensor:
	scalar = torch.zeros_like(STATION_DISTANCES, device=DEVICE)
	for line in individual:
		for a, b in line.edges():
			idx_a = STATION_INDEX.get(a)
			idx_b = STATION_INDEX.get(b)
			if idx_a is None or idx_b is None:
				continue
			scalar[idx_a, idx_b] += 1
			scalar[idx_b, idx_a] += 1
	scalar = scalar.reciprocal()
	return scalar

### Determines minimum path based on scaled distance using D'Esopo-Pape Algorithm
def build_commute_distance_tensor(individual: Sequence[TubeLine]) -> torch.Tensor:
	connectivity = build_connectivity_tensor(individual)
	scaled_connectivity = connectivity * STATION_DISTANCES
	paths = torch.full_like(scaled_connectivity, float(DISCONNECT_MULTIPLIER), device=DEVICE)

	for r, origin in enumerate(paths):
		distance = [float("inf")] * len(origin)
		distance[r] = 0
		in_queue = [False] * len(origin)
		queue = deque()
		in_queue[r] = True
		queue.append(r)
		while queue:
			a = queue.popleft()
			in_queue[a] = False
			for b, destination in enumerate(scaled_connectivity[a]):
				if distance[b] > destination + distance[a]:
					distance[b] = destination + distance[a]
					if not in_queue[b]:
						in_queue[b] = True
						if not queue or distance[b] > distance[queue[0]]:
							queue.append(b)
						else:
							queue.appendleft(b)
		for i, d in enumerate(distance):
			if not math.isinf(d):
				paths[r, i] = d
	# np.savetxt("tensor.csv", paths.cpu().numpy(), delimiter=",")
	return paths

					
			




def evaluate_network(individual: Sequence[TubeLine], demand_tensor: torch.Tensor):
	score_tensor = build_commute_distance_tensor(individual) * demand_tensor


	# np.savetxt("station_distances.csv", STATION_DISTANCES.cpu().numpy(), delimiter=",", fmt="%.6f")
	# np.savetxt("distance_scalar.csv", distance_scalar.cpu().numpy(), delimiter=",", fmt="%.6f")
	# np.savetxt("demand_tensor.csv", demand_tensor.cpu().numpy(), delimiter=",", fmt="%.6f")
	score = score_tensor.sum().item()
	if not math.isfinite(score):
		score = float("inf")
	return (score,)


def configure_toolbox(
	data_path: Path,
	cost_per_connection: int,
	max_cost: int,
):
	stations, demand_tensor = load_station_demands(data_path)
	if not stations:
		raise RuntimeError("No stations found in the supplied matrix")

	station_names = tuple(sorted(stations.keys(), key=lambda name: STATION_INDEX[name]))
	max_cost = float(max_cost)
	cost_per_connection = float(cost_per_connection)

	if "FitnessMin" not in creator.__dict__:
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	if "Individual" not in creator.__dict__:
		creator.create("Individual", list, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	toolbox.register(
		"individual",
		init_individual,
		creator.Individual,
		station_names,
		max_cost,
		cost_per_connection,
	)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", mate_individual, max_cost=max_cost, cost_per_connection=cost_per_connection)
	toolbox.register(
		"mutate",
		mutate_individual,
		station_names=station_names,
		max_cost=max_cost,
		cost_per_connection=cost_per_connection,
		indpb=0.25,
	)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("evaluate", evaluate_network, demand_tensor=demand_tensor)

	return toolbox, station_names


def run_algorithm(
	population_size: int,
	generations: int,
	cxpb: float,
	mutpb: float,
	seed: int | None,
	data_path: Path,
	cost_per_connection: int,
	max_cost: int,
	map_output: Path | None,
	temperature: float = 1.0,
):
	if seed is not None:
		random.seed(seed)

	toolbox, station_names = configure_toolbox(
		data_path=data_path,
		cost_per_connection=cost_per_connection,
		max_cost=max_cost,
	)
	layout = build_layout(station_names)

	population = toolbox.population(n=population_size)
	hof = tools.HallOfFame(10)
	stats = tools.Statistics(lambda ind: ind.fitness.values[0])
	stats.register("avg", statistics.mean)
	stats.register("min", min)
	stats.register("max", max)

	# Custom generational loop with elitism (mu+lambda selection) to preserve best individuals
	logbook = tools.Logbook()
	logbook.header = ["gen", "nevals", "avg", "min", "max"]

	# Evaluate initial population
	invalid = [ind for ind in population if not ind.fitness.valid]
	fitnesses = list(map(toolbox.evaluate, invalid))
	for ind, fit in zip(invalid, fitnesses):
		ind.fitness.values = fit
	hof.update(population)
	record = stats.compile(population)
	logbook.record(gen=0, nevals=len(invalid), **record)
	print(logbook.stream)

	for gen in range(1, generations + 1):
		# Selection
		offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

		# Effective rates are scaled by temperature (temp>1 -> increase rates, temp<1 -> decrease)
		effective_cxpb = max(0.0, min(1.0, cxpb * temperature))
		effective_mutpb = max(0.0, min(1.0, mutpb * temperature))

		# Apply crossover
		for i in range(1, len(offspring), 2):
			if random.random() < effective_cxpb:
				toolbox.mate(offspring[i - 1], offspring[i])
				# mark fitness invalid
				if hasattr(offspring[i - 1], 'fitness'):
					try:
						del offspring[i - 1].fitness.values
					except Exception:
						pass
				if hasattr(offspring[i], 'fitness'):
					try:
						del offspring[i].fitness.values
					except Exception:
						pass

		# Apply mutation
		for mutant in offspring:
			if random.random() < effective_mutpb:
				toolbox.mutate(mutant)
				if hasattr(mutant, 'fitness'):
					try:
						del mutant.fitness.values
					except Exception:
						pass

		# Evaluate invalid individuals
		invalid = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = list(map(toolbox.evaluate, invalid))
		for ind, fit in zip(invalid, fitnesses):
			ind.fitness.values = fit

		# Combine parents and offspring then select the best to form the next generation
		population = tools.selBest(population + offspring, population_size)
		hof.update(population)

		# Record statistics
		record = stats.compile(population)
		logbook.record(gen=gen, nevals=len(invalid), **record)
		print(logbook.stream)

	if hof.items:
		# Print only the top (best) individual from the Hall of Fame
		best = hof.items[0]
		normalized_lines = normalize_line_names(best)
		score = best.fitness.values[0]
		print(f"\nBest individual score: {score}")
		print("  Total connections:", total_connections(normalized_lines))
		for line in normalized_lines:
			print(f"    {line.name}: {' -> '.join(line.stations)}")
		if map_output:
			render_transit_map(normalized_lines, layout, map_output)
	else:
		print("No valid individuals generated.")

	return population, logbook, hof


def normalize_line_names(lines: Sequence[TubeLine]) -> List[TubeLine]:
	"""Return copies of lines renamed sequentially (L1, L2, ...) without clone suffixes."""

	normalized: List[TubeLine] = []
	for idx, line in enumerate(lines, start=1):
		new_line = TubeLine(name=f"L{idx}", stations=list(line.stations))
		normalized.append(new_line)
	return normalized


def _edge_key(a: str, b: str) -> Tuple[str, str]:
	return tuple(sorted((a, b)))


def compute_edge_offsets(lines: Sequence[TubeLine]) -> Dict[Tuple[Tuple[str, str], int], float]:
	"""Assign offset values for lines that share identical station-to-station segments."""

	usage: Dict[Tuple[str, str], List[int]] = defaultdict(list)
	for idx, line in enumerate(lines):
		for a, b in zip(line.stations[:-1], line.stations[1:]):
			usage[_edge_key(a, b)].append(idx)

	offsets: Dict[Tuple[Tuple[str, str], int], float] = {}
	for key, line_indices in usage.items():
		if len(line_indices) == 1:
			offsets[(key, line_indices[0])] = 0.0
			continue
		indices = sorted(line_indices)
		count = len(indices)
		for order, line_idx in enumerate(indices):
			shift = (order - (count - 1) / 2) * PARALLEL_OFFSET
			offsets[(key, line_idx)] = shift
	return offsets


def draw_station_node(ax, pos: Tuple[float, float], name: str, served: bool) -> None:
	"""Draw a station marker and label, differentiating served vs. unserved nodes."""
	size = 60 if served else 46
	face = "white" if served else "#ededed"
	edge = "#2b2b2b" if served else "#a0a0a0"
	text_color = "#1a1a1a" if served else "#6b6b6b"
	zorder = 4 if served else 3.2
	ax.scatter(
		pos[0],
		pos[1],
		s=size,
		facecolors=face,
		edgecolors=edge,
		linewidths=1.5,
		zorder=zorder,
	)
	ax.text(
		pos[0] + 0.1,
		pos[1] + 0.1,
		name,
		fontsize=7,
		fontweight="medium",
		color=text_color,
		zorder=zorder + 0.3,
	)


def geo_octilinear_route(start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], List[bool]]:
	"""Return a geo-octilinear (0°/45°/90°/135°) polyline between two coordinates."""

	def buffered_segment(
		p0: Tuple[float, float],
		p1: Tuple[float, float],
		start_flag: bool,
		end_flag: bool,
	) -> Tuple[List[Tuple[float, float]], List[bool]]:
		dx = p1[0] - p0[0]
		dy = p1[1] - p0[1]
		dist = math.hypot(dx, dy)
		if dist == 0:
			return [p0, p1], [start_flag, end_flag]
		dir_x = dx / dist
		dir_y = dy / dist
		gap = min(STATION_BUFFER, dist / 3)
		if gap <= 0:
			return [p0, p1], [start_flag, end_flag]
		inner_start = (p0[0] + dir_x * gap, p0[1] + dir_y * gap)
		inner_end = (p1[0] - dir_x * gap, p1[1] - dir_y * gap)
		return [p0, inner_start, inner_end, p1], [start_flag, False, False, end_flag]

	def add_point(path: List[Tuple[float, float]], target: Tuple[float, float]) -> None:
		if not path or math.hypot(path[-1][0] - target[0], path[-1][1] - target[1]) < 1e-6:
			return
		path.append(target)

	x0, y0 = start
	x1, y1 = end
	dx = x1 - x0
	dy = y1 - y0
	points: List[Tuple[float, float]] = [start]

	abs_dx = abs(dx)
	abs_dy = abs(dy)
	sign_x = 1 if dx >= 0 else -1
	sign_y = 1 if dy >= 0 else -1
	shared = min(abs_dx, abs_dy)
	remaining_x = abs_dx - shared
	remaining_y = abs_dy - shared

	if shared > 1e-6:
		diag_point = (x0 + sign_x * shared, y0 + sign_y * shared)
		add_point(points, diag_point)
	if remaining_x > 1e-6:
		x_target = points[-1][0] + sign_x * remaining_x
		add_point(points, (x_target, points[-1][1]))
	if remaining_y > 1e-6:
		y_target = points[-1][1] + sign_y * remaining_y
		add_point(points, (points[-1][0], y_target))
	if math.hypot(points[-1][0] - end[0], points[-1][1] - end[1]) > 1e-6:
		points.append(end)

	routed_points: List[Tuple[float, float]] = []
	flags: List[bool] = []
	for idx in range(1, len(points)):
		segment_points, segment_flags = buffered_segment(points[idx - 1], points[idx], idx == 1, idx == len(points) - 1)
		if not routed_points:
			routed_points.extend(segment_points)
			flags.extend(segment_flags)
		else:
			routed_points.extend(segment_points[1:])
			flags.extend(segment_flags[1:])
	return routed_points, flags


def apply_segment_offsets(
	points: List[Tuple[float, float]],
	point_is_station: List[bool],
	segment_offsets: List[float],
) -> List[Tuple[float, float]]:
	"""Shift intermediate points away from shared tracks while keeping stations fixed."""

	if not segment_offsets:
		return points
	normals: List[Tuple[float, float]] = []
	for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
		dx = x1 - x0
		dy = y1 - y0
		length = math.hypot(dx, dy)
		if length == 0:
			normals.append((0.0, 0.0))
		else:
			normals.append((-dy / length, dx / length))

	adjusted: List[Tuple[float, float]] = [points[0]]
	for idx in range(1, len(points) - 1):
		if point_is_station[idx]:
			adjusted.append(points[idx])
			continue
		contribs: List[Tuple[float, float]] = []
		prev_offset = segment_offsets[idx - 1]
		prev_norm = normals[idx - 1]
		if prev_offset:
			contribs.append((prev_norm[0] * prev_offset, prev_norm[1] * prev_offset))
		next_offset = segment_offsets[idx]
		next_norm = normals[idx]
		if next_offset:
			contribs.append((next_norm[0] * next_offset, next_norm[1] * next_offset))
		if contribs:
			avg_x = sum(v[0] for v in contribs) / len(contribs)
			avg_y = sum(v[1] for v in contribs) / len(contribs)
			adjusted.append((points[idx][0] + avg_x, points[idx][1] + avg_y))
		else:
			adjusted.append(points[idx])
	adjusted.append(points[-1])
	return adjusted


def rounded_path(
	points: Sequence[Tuple[float, float]],
	point_flags: Sequence[bool],
	radius: float,
) -> MplPath | None:
	"""Create a matplotlib Path with rounded corners from a polyline."""

	if len(points) < 2:
		return None
	verts: List[Tuple[float, float]] = [points[0]]
	codes: List[int] = [MplPath.MOVETO]

	for idx in range(1, len(points) - 1):
		prev_pt = points[idx - 1]
		cur_pt = points[idx]
		next_pt = points[idx + 1]
		dx1 = cur_pt[0] - prev_pt[0]
		dy1 = cur_pt[1] - prev_pt[1]
		dx2 = next_pt[0] - cur_pt[0]
		dy2 = next_pt[1] - cur_pt[1]
		len1 = math.hypot(dx1, dy1)
		len2 = math.hypot(dx2, dy2)
		if len1 == 0 or len2 == 0:
			verts.append(cur_pt)
			codes.append(MplPath.LINETO)
			continue
		ux1, uy1 = dx1 / len1, dy1 / len1
		ux2, uy2 = dx2 / len2, dy2 / len2
		if math.isclose(ux1, -ux2, abs_tol=1e-6) and math.isclose(uy1, -uy2, abs_tol=1e-6):
			verts.append(cur_pt)
			codes.append(MplPath.LINETO)
			continue
		corner_r = min(radius, len1 / 2, len2 / 2)
		if point_flags[idx]:
			clearance = STATION_BUFFER * 0.6
			corner_r = min(
				corner_r,
				max(len1 - clearance, 0) / 2,
				max(len2 - clearance, 0) / 2,
			)
		if corner_r <= 1e-4:
			verts.append(cur_pt)
			codes.append(MplPath.LINETO)
			continue
		entry = (cur_pt[0] - ux1 * corner_r, cur_pt[1] - uy1 * corner_r)
		exit = (cur_pt[0] + ux2 * corner_r, cur_pt[1] + uy2 * corner_r)
		verts.append(entry)
		codes.append(MplPath.LINETO)
		verts.append(cur_pt)
		codes.append(MplPath.CURVE3)
		verts.append(exit)
		codes.append(MplPath.CURVE3)

	verts.append(points[-1])
	codes.append(MplPath.LINETO)
	return MplPath(verts, codes)


def render_transit_map(lines: Sequence[TubeLine], layout: Dict[str, Tuple[float, float]], output_path: Path) -> None:
	"""Render a stylized tube map for the supplied lines using matplotlib."""

	if not lines:
		return

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(11.7, 8.3))  # Roughly horizontal A4 proportions
	fig.patch.set_facecolor("white")
	ax.set_facecolor("white")
	ax.set_aspect("equal")
	ax.axis("off")

	color_cycle = plt.rcParams.get("axes.prop_cycle", None)
	colors = color_cycle.by_key()["color"] if color_cycle else [
		"#0078bf",
		"#e32017",
		"#ffd300",
		"#0098d4",
		"#95c11f",
		"#003688",
		"#f386a1",
		"#a0a5a7",
		"#009d4c",
		"#d799af",
	]

	station_line_counts: Dict[str, int] = defaultdict(int)
	for line in lines:
		for name in line.stations:
			station_line_counts[name] += 1

	edge_offsets = compute_edge_offsets(lines)

	for idx, line in enumerate(lines):
		path = build_line_path(line, layout, idx, edge_offsets)
		if path is None:
			continue
		color = colors[idx % len(colors)]
		patch = mpatches.PathPatch(
			path,
			facecolor="none",
			edgecolor=color,
			linewidth=LINE_WIDTH_PT,
			capstyle="round",
			joinstyle="round",
			alpha=0.97,
			zorder=2,
		)
		ax.add_patch(patch)

	drawn: set[str] = set()
	for line in lines:
		for name in line.stations:
			if name in drawn:
				continue
			pos = layout.get(name)
			if not pos:
				continue
			drawn.add(name)
			served = station_line_counts.get(name, 0) > 0
			draw_station_node(ax, pos, name, served)

	for name, pos in layout.items():
		if name in drawn:
			continue
		draw_station_node(ax, pos, name, served=False)

	ax.relim()
	ax.autoscale_view()

	format_hint = output_path.suffix.lower().lstrip(".") or "svg"
	fig.savefig(output_path, format=format_hint, dpi=300, bbox_inches="tight", facecolor="white")
	plt.close(fig)
	print(f"Saved transit map to {output_path}")


def build_line_path(
	line: TubeLine,
	layout: Dict[str, Tuple[float, float]],
	line_idx: int,
	edge_offsets: Dict[Tuple[Tuple[str, str], int], float],
) -> MplPath | None:
	"""Construct a rounded matplotlib path for a tube line with offsets applied."""

	if len(line.stations) < 2:
		return None
	points: List[Tuple[float, float]] = []
	point_flags: List[bool] = []
	segment_offsets: List[float] = []

	first = layout.get(line.stations[0])
	if first is None:
		return None
	points.append(first)
	point_flags.append(True)

	for idx in range(1, len(line.stations)):
		start_name = line.stations[idx - 1]
		end_name = line.stations[idx]
		start = layout.get(start_name)
		end = layout.get(end_name)
		if start is None or end is None:
			return None
		segment_points, flags = geo_octilinear_route(start, end)
		key = _edge_key(start_name, end_name)
		offset = edge_offsets.get((key, line_idx), 0.0)
		for point_idx in range(1, len(segment_points)):
			points.append(segment_points[point_idx])
			point_flags.append(flags[point_idx])
			segment_offsets.append(offset)

	if len(points) < 2:
		return None
	adjusted = apply_segment_offsets(points, point_flags, segment_offsets)
	path = rounded_path(adjusted, point_flags, radius=CORNER_RADIUS)
	return path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Optimize tube connections using DEAP")
	parser.add_argument("--data", type=Path, default=Path("filtered_OD.csv"), help="Path to the filtered OD CSV")
	parser.add_argument("--pop-size", type=int, default=20, help="Population size")
	parser.add_argument("--generations", type=int, default=5, help="Number of generations to evolve")
	parser.add_argument("--cxpb", type=float, default=0.5, help="Crossover probability")
	parser.add_argument("--mutpb", type=float, default=0.3, help="Mutation probability")
	parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1), help="Random seed for reproducibility")
	parser.add_argument(
		"--cost-per-connection",
		type=int,
		default=COST_PER_CONNECTION_DEFAULT,
		help="Cost assigned to each station-to-station connection",
	)
	parser.add_argument(
		"--max-cost",
		type=int,
		default=MAX_COST_DEFAULT,
		help="Maximum total cost allowed for an individual",
	)
	parser.add_argument(
		"--map-output",
		type=str,
		default="optimized_map.svg",
		help="Path to save the transit map (SVG/PDF). Use '' to skip rendering.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="Temperature scalar to scale crossover/mutation rates (>1 increases, <1 decreases)",
	)
	
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	print(f"Using random seed: {args.seed}")
	map_output: Path | None
	if args.map_output and args.map_output.strip():
		map_output = Path(args.map_output)
	else:
		map_output = None

	run_algorithm(
		population_size=args.pop_size,
		generations=args.generations,
		cxpb=args.cxpb,
		mutpb=args.mutpb,
		seed=args.seed,
		data_path=args.data,
		cost_per_connection=args.cost_per_connection,
		max_cost=args.max_cost,
		map_output=map_output,
		temperature=args.temperature,
	)


if __name__ == "__main__":
	main()

