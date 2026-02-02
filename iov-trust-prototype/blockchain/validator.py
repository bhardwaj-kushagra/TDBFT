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

def check_consensus_weighted(committee: List, proposal_score=1.0) -> bool:
    """
    Abstracted Consensus (Section IV-B).
    Calculates weighted votes.
    
    Block accepted if Sum(ti * si) >= (2/3) * Sum(ti).
    Here assuming si (vote) is 1.0 (Correct) if vehicle is HONEST, 
    and 0.0 (Reject/Bad) if MALICIOUS/SWING?
    
    Actually, Malicious nodes might vote NO or PROPOSE BAD BLOCKS.
    Assuming this function checks if a block proposed by Leader is accepted.
    """
    total_trust_mass = sum(v.global_trust_score for v in committee)
    if total_trust_mass == 0:
        return False
        
    approval_mass = 0.0
    
    for member in committee:
        # Honest members vote YES (approving a theoretically good proposal)
        # Malicious members vote NO (trying to stall or they propose bad blocks)
        # Simplified: Honest = Vote 1, Malicious = Vote 0
        
        # Real PBFT: 2/3 majority.
        # Weighted PBFT: 2/3 of Trust Mass.
        
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
        # Honest = Vote 1, Malicious = Vote 0
        votes += 1 if not member.is_malicious else 0
        
    threshold = (2.0 / 3.0) * n
    return votes >= threshold
