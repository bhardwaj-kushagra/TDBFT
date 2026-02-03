"""
Mocked DAG (Directed Acyclic Graph) Blockchain.

Stores trust updates.
Not a real blockchain (no PoW/PoS, no hashing).
"""
from typing import List, Dict
import random
from .block import Block

class DAG:
    def __init__(self):
        self.blocks: Dict[str, Block] = {}
        # Simple tips management: keep track of blocks that have no children (simplified)
        # For this prototype, we just link to the last N blocks added.
        self.tips: List[str] = [] 

    def add_block(self, data, validator_id, issuer_trust=0.0, n_parents: int = 2) -> Block:
        """
        Creates and adds a new block to the DAG.
        Uses Trust-Aware Tip Selection (Suggestion B).
        Updates TCW recursively (Suggestion A).
        """
        # Tip Selection: Select n_parents from self.tips using Trust-Aware Selection
        parents = self.select_tips_trust_aware(n_parents)
        
        new_block = Block(data, validator_id, parents, issuer_trust)
        self.blocks[new_block.id] = new_block
        
        # Update tips:
        for p_id in parents:
            if p_id in self.tips:
                self.tips.remove(p_id)
        
        self.tips.append(new_block.id)
        
        # Propagate TCW (Suggestion A)
        # TCW(b) = Trust(Issuer) + Sum(TCW(children))
        # This implies adding new_block.tcw (which is issuer_trust) to all ancestors
        self.propagate_tcw(parents, new_block.tcw)
        
        return new_block

    def select_tips_trust_aware(self, n_parents: int) -> List[str]:
        """
        Implements Trust-Aware Tip Selection (Equation 27 - inferred).
        Preferentially selects tips with higher TCW or Issuer Trust.
        """
        if not self.tips:
            return []
            
        if len(self.tips) <= n_parents:
            return list(self.tips)
            
        # Get Tip objects
        tip_blocks = [self.blocks[tid] for tid in self.tips]
        
        # Calculate weights based on TCW (or issuer trust if TCW is not built up yet)
        # Using simple proportional weight
        weights = [b.tcw + 1e-9 for b in tip_blocks] # +epsilon to avoid zero division
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Weighted random selection
        # Note: random.choices returns with replacement, simplified logic uses sample if no weights
        # We manually implement weighted sample without replacement is tricky, 
        # so we just pick top-N for deterministic behavior or use choices and distinct.
        
        # For prototype: Just pick the heaviest tips (Greedy) or Weighted?
        # "Preferentially reference" implies probability.
        # Let's use random.choices and dedup.
        
        selected = set()
        while len(selected) < n_parents:
            pick = random.choices(tip_blocks, weights=weights, k=1)[0]
            selected.add(pick.id)
            
        return list(selected)

    def propagate_tcw(self, parent_ids: List[str], delta: float):
        """
        Propagates the Trust Weight (delta) upwards to ancestors.
        TCW(parent) += delta
        """
        # Using a queue for BFS/DFS propagation
        # Note: The user formula "Sum TCW(child)" implies we count support from ALL paths.
        # So we do NOT use a visited set to prune distinct nodes, 
        # BUT strictly we should be careful about exponential blowup.
        # Given "narrow DAG", we'll allow path-based summation.
        
        queue = list(parent_ids)
        # However, to prevent absolute infinite loops if cycles exist (bug safety), we limit depth or strict DAG.
        # Since it's a DAG, no cycles. 
        # Optim: If block structure is simple, list growth is minimal.
        
        processed_count = 0
        limit = 10000 # Safety break
        
        while queue and processed_count < limit:
            curr_id = queue.pop(0)
            if curr_id in self.blocks:
                block = self.blocks[curr_id]
                block.tcw += delta
                queue.extend(block.parents)
                processed_count += 1

    def is_finalized(self, block_id: str, threshold: float) -> bool:
        """
        Suggestion C: Check Finality
        Finalized if TCW(block) >= Theta
        """
        if block_id not in self.blocks:
            return False
        return self.blocks[block_id].tcw >= threshold

    def get_history(self):
        return self.blocks.values()

    def merge_with(self, other_dag):
        """
        Merges blocks from another DAG into this one.
        Resolves tips.
        """
        for block_id, block in other_dag.blocks.items():
            if block_id not in self.blocks:
                self.blocks[block_id] = block
        
        # Merge tips: Union of tips
        # In a real DAG this is complex (checking ancestry).
        # Simulated: Just ensure all tips are tracked.
        # But for simplicity, let's just append the other tips to our tips
        # and keep the last 5 to avoid explosion.
        
        combined_tips = list(set(self.tips + other_dag.tips))
        self.tips = combined_tips[-5:] # Keep latest 5 tips

