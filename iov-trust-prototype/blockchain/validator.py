"""
Validator Selection Logic.

Selects top-N trusted vehicles to be validators.
"""
from typing import List

def select_validators(ranked_vehicles: List, top_n: int = 3) -> List:
    """
    Selects the top N vehicles from the ranked list to act as validators.
    
    Args:
        ranked_vehicles: List of Vehicle objects (assumed sorted by trust desc).
        top_n: Number of validators to select.
        
    Returns:
        List of Vehicle objects that are chosen.
    """
    # Simply slice the top N
    # In a real system, we might check for min_trust_threshold.
    
    candidates = ranked_vehicles[:top_n]
    
    # Filter out low trust if needed (e.g. must be > 0.5)
    validators = [v for v in candidates if v.global_trust_score > 0.5]
    
    # If not enough high-trust vehicles, we might return fewer or none.
    return validators
