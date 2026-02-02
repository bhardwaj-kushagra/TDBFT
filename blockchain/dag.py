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

    def add_block(self, data, validator_id, n_parents: int = 2) -> Block:
        """
        Creates and adds a new block to the DAG.
        Uses Tip Selection Algorithm (random selection of n_parents from current tips).
        """
        # Tip Selection: Select n_parents from self.tips
        available_tips = self.tips
        if not available_tips:
            parents = []
        elif len(available_tips) <= n_parents:
            parents = list(available_tips)
        else:
            parents = random.sample(available_tips, n_parents)
        
        new_block = Block(data, validator_id, parents)
        self.blocks[new_block.id] = new_block
        
        # Update tips:
        # Referenced parents are no longer tips (they are now confirmed/covered).
        # The new block becomes a tip.
        # Tips that were not selected remain tips (this creates/preserves DAG width).
        for p_id in parents:
            if p_id in self.tips:
                self.tips.remove(p_id)
        
        self.tips.append(new_block.id)
        
        return new_block

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

