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
    
    for v in committee:
        # Determine vote behavior
        if v.behavior_type == 'HONEST':
            vote = 1 # Approve valid block
        else:
            vote = 0 # Reject valid block (DoS) OR Approve bad block
            
        # Weighted vote
        approval_mass += v.global_trust_score * vote
        
    # 2/3rds majority of TRUST, not nodes
    return approval_mass >= (2/3 * total_trust_mass)
