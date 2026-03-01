"""
Validator Selection Logic.

Selects top-N trusted vehicles to be validators.
Implements Section IV-A (Committee Selection).
"""
from typing import List

def select_validators(ranked_vehicles: List, top_n: int = 3) -> List:
    """
    Selects the top C vehicles to form the consensus committee.
    
    Args:
        ranked_vehicles: List of Vehicle objects (assumed sorted by trust desc).
        top_n: Committee size (c).
        
    Returns:
        List of Vehicle objects that are chosen.
        Element 0 is the Leader (max trust).
    """
    # Simple slice for top-c (Section IV-A)
    # The list is already sorted by Global Trust from the TrustModel
    
    committee = ranked_vehicles[:top_n]
    
    # Optional logic: Byzantine Check
    # Ensure all selected have reasonably high trust
    # If leaders have trust < 0.5, maybe we shouldn't act?
    # For prototype, we keep them but maybe flag.
    
    return committee

def check_consensus_weighted(committee: List) -> bool:
    """
    Abstracted Consensus (Section IV-B).
    Calculates weighted votes.
    
    Block accepted if Sum(ti * si) >= (2/3) * Sum(ti).
    Honest nodes vote YES, malicious nodes vote NO.
    Swing nodes vote based on their current phase.
    """
    total_trust_mass = sum(v.global_trust_score for v in committee)
    if total_trust_mass == 0:
        return False
        
    approval_mass = 0.0
    
    for member in committee:
        # Phase-aware voting:
        # Honest members always vote YES.
        # Malicious members always vote NO.
        # Swing members vote YES in good phase, NO in bad phase.
        if hasattr(member, 'is_in_good_phase'):
            vote = 1.0 if member.is_in_good_phase() else 0.0
        else:
            vote = 1.0 if not member.is_malicious else 0.0
        approval_mass += vote * member.global_trust_score
        
    threshold = (2.0 / 3.0) * total_trust_mass
    return approval_mass >= threshold

def check_consensus_simple(committee: List) -> bool:
    """
    Standard PBFT Consensus (1 Node = 1 Vote).
    Used for LT-PBFT baseline.
    Block accepted if Votes > 2/3 of Committee Size.
    """
    n = len(committee)
    if n == 0: return False
    
    votes = 0
    for member in committee:
        # Phase-aware voting: swing nodes cooperate during good phase
        if hasattr(member, 'is_in_good_phase'):
            votes += 1 if member.is_in_good_phase() else 0
        else:
            votes += 1 if not member.is_malicious else 0
        
    threshold = (2.0 / 3.0) * n
    return votes >= threshold
