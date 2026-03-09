# Configuration file for high-level framework parameters

DIMENSION = 42
NUM_BOND_TYPES = 5

# Context norms for 15-39 model
CONTEXT_NORMS = {
    "mean": [105.0766, 473.1938, 537.4675],
    "mad": [52.0409, 219.7475, 232.9718],
}

# Context norms for 6-39 model
CONTEXT_NORMS_NEW = {
    'mean': [92.1179, 413.7875, 470.5253],
    'mad': [53.4742, 232.5775, 251.1813],
}
ATOM_DECODER = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "P",
    5: "S",
    6: "Cl",
    7: "Br",
}

PERMITTED_ELEMENTS = (
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    35,
)

MIN_N_NODES = 6
MAX_N_NODES = 39
