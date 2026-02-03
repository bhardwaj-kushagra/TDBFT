"""
Consensus Manager Module.

Abstracts the blockchain consensus process:
1. Validator Selection (Committee formation).
2. Consensus Execution (Voting/PBFT).
3. Block Creation and DAG Appending.

This removes duplicate logic from the experiment runners.
"""
from typing import List, Dict
from blockchain.validator import select_validators, check_consensus_weighted, check_consensus_simple
from blockchain.dag import DAG
from blockchain.block import Block

class ConsensusManager:
    def __init__(self, model_type: str, vehicles: Dict, dag: DAG = None):
        """
        Args:
            model_type: The trust model being used (determines voting logic).
            vehicles: Dictionary of all vehicle objects {id: Vehicle}.
            dag: The blockchain DAG instance (optional, creates one if None).
        """
        self.model_type = model_type
        self.vehicles = vehicles
        self.dag = dag if dag else DAG()
        
        self.consensus_success_count = 0
        self.total_consensus_rounds = 0

    def attempt_consensus(self, step: int, committee_size: int = 5) -> bool:
        """
        Runs one round of consensus.
        
        1. Selects top validators based on global trust.
        2. Checks if committee reaches consensus.
        3. Appends block if successful.
        
        Returns:
            bool: True if consensus reached, False otherwise.
        """
        self.total_consensus_rounds += 1
        
        # 1. Select Validators
        # Currently, simplest logic: sort by global trust
        ranked_vehicles = sorted(
            self.vehicles.values(), 
            key=lambda v: v.global_trust_score, 
            reverse=True
        )
        
        committee = select_validators(ranked_vehicles, top_n=committee_size)
        
        if not committee:
            return False

        # 2. Check Consensus
        passed = False
        if self.model_type in ['LT_PBFT', 'COBATS']:
            # Simple Majority for traditional BFT-style models
            passed = check_consensus_simple(committee) 
        else:
            # Weighted Voting for Trust-based models (PROPOSED, BTVR, etc.)
            passed = check_consensus_weighted(committee)
            
        # 3. Append to DAG if passed
        if passed:
            self.consensus_success_count += 1
            
            # Create a block containing current trust snapshots
            trust_snapshot = {v.id: v.global_trust_score for v in self.vehicles.values()}
            
            # Use DAG's internal block creation logic
            # committee[0] is the primary validator/leader for this block
            # Suggestion A: Assign Issuer Trust for TCW calculation
            leader = committee[0]
            self.dag.add_block(
                data=trust_snapshot, 
                validator_id=leader.id, 
                issuer_trust=leader.global_trust_score
            )
            
        return passed
