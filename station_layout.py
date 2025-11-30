"""Representative station coordinates for stylized transit map rendering."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple
import torch

# Coordinates are intentionally schematic (not geographic) to resemble a tube-style map.
# X increases eastward, Y increases northward.
MIN_STATION_SPACING = 40
STATION_LAYOUT: Dict[str, Tuple[float, float]] = {
    "Paddington": (-6.5, 1.4),
    "Edgware Road (Cir)": (-6.2, 0.8),
    "Edgware Road (Bak)": (-6.2, 0.3),
    "Notting Hill Gate": (-7.0, 0.1),
    "Bayswater": (-6.6, 0.5),
    "Queensway": (-6.8, 0.3),
    "High Street Kensington": (-6.2, -0.4),
    "Kensington (Olympia)": (-6.4, -1.0),
    "Gloucester Road": (-5.4, -0.6),
    "South Kensington": (-5.0, -0.6),
    "Earl's Court": (-5.2, -1.5),
    "West Brompton": (-5.4, -2.0),
    "Sloane Square": (-4.0, -1.2),
    "Victoria": (-3.4, -1.5),
    "Pimlico": (-2.8, -2.1),
    "Vauxhall": (-2.2, -2.4),
    "Battersea Power Station": (-2.4, -3.0),
    "Hyde Park Corner": (-3.8, -0.4),
    "Green Park": (-2.6, -0.3),
    "Bond Street": (-2.1, -0.1),
    "Marble Arch": (-2.9, 0.1),
    "Oxford Circus": (-1.6, -0.1),
    "Piccadilly Circus": (-1.3, -0.5),
    "Charing Cross": (-0.7, -0.9),
    "Embankment": (-0.5, -1.2),
    "Westminster": (-0.6, -1.5),
    "St. James's Park": (-0.9, -1.3),
    "Waterloo": (0.0, -1.7),
    "Lambeth North": (-0.2, -2.0),
    "Elephant & Castle": (0.3, -2.3),
    "Kennington": (0.5, -2.6),
    "Borough": (0.7, -1.8),
    "London Bridge": (1.0, -1.6),
    "Southwark": (0.5, -1.9),
    "Blackfriars": (0.8, -1.2),
    "Temple": (0.2, -1.0),
    "Holborn": (-0.4, 0.0),
    "Russell Square": (-0.2, 0.6),
    "Tottenham Court Road": (-0.6, 0.2),
    "Goodge Street": (-0.8, 0.5),
    "Euston Square": (-0.6, 1.2),
    "Euston": (-0.4, 1.5),
    "Warren Street": (-0.8, 1.0),
    "Regent's Park": (-1.0, 0.8),
    "Great Portland Street": (-0.4, 0.9),
    "Baker Street": (-1.6, 0.9),
    "Marylebone": (-1.8, 1.2),
    "Edgware Road": (-2.2, 1.0),
    "King's Cross St. Pancras": (0.4, 1.4),
    "St. Pancras International": (0.5, 1.6),
    "Angel": (1.5, 1.4),
    "Old Street": (1.4, 0.9),
    "Barbican": (1.0, 0.5),
    "Farringdon": (0.8, 0.3),
    "Chancery Lane": (0.4, 0.0),
    "Covent Garden": (-0.2, -0.2),
    "Leicester Square": (-0.4, -0.4),
    "Mansion House": (1.4, -0.9),
    "Cannon Street": (1.6, -1.0),
    "Bank / Monument": (1.8, -0.7),
    "Bank": (1.8, -0.6),
    "Monument": (2.0, -0.8),
    "St. Paul's": (1.3, -0.3),
    "Moorgate": (1.3, 0.5),
    "Liverpool Street": (1.8, 0.6),
    "Shoreditch High Street": (2.4, 1.0),
    "Hoxton": (2.6, 1.2),
    "Aldgate": (2.4, -0.2),
    "Aldgate East": (2.6, 0.0),
    "Tower Hill": (2.4, -0.6),
    "Tower Gateway": (2.6, -0.7),
    "Fenchurch Street": (2.2, -0.5),
    "Blackfriars": (0.8, -1.2),
    "Bermondsey": (1.4, -2.0),
    "Canada Water": (1.8, -2.2),
    "Wapping": (2.3, -1.6),
    "Whitechapel": (2.9, 0.3),
    "Bethnal Green": (3.2, 0.6),
    "Mile End": (3.5, 0.2),
}

# Preserve the same ordering as the literal above
STATION_ORDER: Tuple[str, ...] = tuple(STATION_LAYOUT.keys())

_n = len(STATION_ORDER)
_dist_matrix = [[0.0] * _n for _ in range(_n)]
for i, a in enumerate(STATION_ORDER):
    xa, ya = STATION_LAYOUT[a]
    for j, b in enumerate(STATION_ORDER):
        if i == j:
            _dist_matrix[i][j] = 0.0
            continue
        xb, yb = STATION_LAYOUT[b]
        # Euclidean distance in schematic units, scaled to meters
        _dist_matrix[i][j] = math.hypot(xa - xb, ya - yb)

# Exported constant: float32 PyTorch tensor of shape (N, N)
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
DEVICE = "cpu"
STATION_DISTANCES = torch.tensor(_dist_matrix, dtype=torch.float32, device=DEVICE)


def build_layout(stations: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Return coordinates for every requested station, fabricating a grid for unknown names."""

    layout: Dict[str, Tuple[float, float]] = {}
    for name in stations:
        if name in STATION_LAYOUT:
            layout[name] = STATION_LAYOUT[name]

    missing = [name for name in stations if name not in layout]
    if missing:
        cols = max(4, math.ceil(math.sqrt(len(missing))))
        spacing = 1.2
        start_x = 5.0
        start_y = 3.0
        for idx, name in enumerate(sorted(missing)):
            col = idx % cols
            row = idx // cols
            layout[name] = (start_x + col * spacing, start_y - row * spacing)

    return enforce_min_spacing(layout, MIN_STATION_SPACING)


def enforce_min_spacing(
    layout: Dict[str, Tuple[float, float]],
    min_distance: float,
    iterations: int = 40,
    step: float = 0.35,
) -> Dict[str, Tuple[float, float]]:
    """Gently spreads stations apart so no two are closer than ``min_distance``."""

    coords = {name: [x, y] for name, (x, y) in layout.items()}
    items = list(coords.items())
    for _ in range(iterations):
        moved = False
        for i in range(len(items)):
            name_i, pos_i = items[i]
            for j in range(i + 1, len(items)):
                name_j, pos_j = items[j]
                dx = pos_j[0] - pos_i[0]
                dy = pos_j[1] - pos_i[1]
                dist = math.hypot(dx, dy)
                if dist == 0:
                    dx, dy, dist = 0.01, 0.0, 0.01
                if dist < min_distance:
                    overlap = (min_distance - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    shift = overlap * step
                    pos_i[0] -= nx * shift
                    pos_i[1] -= ny * shift
                    pos_j[0] += nx * shift
                    pos_j[1] += ny * shift
                    moved = True
        if not moved:
            break

    return {name: (pos[0], pos[1]) for name, pos in coords.items()}

