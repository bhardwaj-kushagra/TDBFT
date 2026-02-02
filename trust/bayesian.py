"""
Bayesian Trust Computation Module.

This module implements the Beta reputation system.
Trust = alpha / (alpha + beta)
"""

def compute_trust(alpha: float, beta: float) -> float:
    """
    Computes the Local Trust Value (Section III-A).
    Formula: mij = (yij + a) / (nij + a + b)
    Where a=1, b=1.
    
    Args:
        alpha (float): Positive interactions + 1. (corresponds to yij + a)
        beta (float): Negative interactions + 1. (corresponds to nij - yij + b)
        
    Returns:
        float: Trust value between 0.0 and 1.0.
    """
    if (alpha + beta) == 0:
        return 0.5  # Default/Uncertain
    return alpha / (alpha + beta)

def update_parameters(current_alpha: float, current_beta: float, 
                      is_positive: bool, weight: float = 1.0) -> tuple:
    """
    Updates alpha and beta parameters based on a new interaction.
    Matches the accumulation of yij and nij in Section III-A.
    
    Args:
        current_alpha (float): Current alpha value.
        current_beta (float): Current beta value.
        is_positive (bool): True if interaction was good, False if bad.
        weight (float): Weight of the interaction (default=1.0).
        
    Returns:
        tuple: (new_alpha, new_beta)
    """
    if is_positive:
        return current_alpha + weight, current_beta
    else:
        return current_alpha, current_beta + weight
