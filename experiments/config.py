"""
Configuration Module.

Centralizes all simulation constants, model definitions, and visual styles.
This ensures consistency across the non-SUMO and SUMO experiments and the plotting modules.
"""
import os

# ==========================================
# SIMULATION SETTINGS
# ==========================================
DEFAULT_NUM_VEHICLES = 50
DEFAULT_MALICIOUS_PERCENT = 0.1
DEFAULT_SWING_PERCENT = 0.05
DEFAULT_RSUS = 2

# Interactions
DEFAULT_STEPS = 100
INTERACTIONS_PER_STEP_RATIO = 0.5  # If N=50, interactions=25 per step
DEFAULT_ATTACK_INTENSITY = 0.8     # Probability of bad behavior for malicious node

# Trust Logic
SWING_CYCLE_LENGTH = 50            # How many steps a swing attacker behaves nicely/badly
DETECTION_THRESHOLD = 0.35         # Normalized trust score below which a node is flagged

# SUMO Specific
INTERACTION_RANGE = 100.0          # Meters
INTERACTION_PROBABILITY = 0.3      # Chance to interact if in range
SUMO_STEPS = 500
SUMO_VEHICLE_COUNT = 20

# ==========================================
# MODELS & VISUALS
# ==========================================
MODELS = ['BTVR', 'BSED', 'RTM', 'COBATS', 'LT_PBFT', 'PROPOSED']

# Plotting Styles
COLORS = {
    'BTVR': 'blue',
    'BSED': 'green',
    'RTM': 'orange',
    'COBATS': 'cyan',
    'LT_PBFT': 'purple',
    'PROPOSED': 'red'
}

LINE_STYLES = {
    'BTVR': '--',
    'BSED': '--',
    'RTM': '--',
    'COBATS': '--',
    'LT_PBFT': '--',
    'PROPOSED': '-'
}

LINE_WIDTHS = {
    'PROPOSED': 3,
    'DEFAULT': 1.5
}

def get_style(model):
    """Returns a dict of matplotlib style arguments for a given model."""
    return {
        'color': COLORS.get(model, 'black'),
        'linestyle': LINE_STYLES.get(model, '-'),
        'linewidth': LINE_WIDTHS.get(model, LINE_WIDTHS['DEFAULT']),
        'label': model
    }

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
SUMO_CONFIG_PATH = os.path.join(BASE_DIR, '..', 'sumo', 'config.sumocfg')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
