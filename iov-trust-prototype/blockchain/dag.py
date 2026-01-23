"""
Mocked DAG (Directed Acyclic Graph) Blockchain.

Stores trust updates.
Not a real blockchain (no PoW/PoS, no hashing).
"""
from typing import List, Dict
from .block import Block

class DAG:
    def __init__(self):
        self.blocks: Dict[str, Block] = {}
        # Simple tips management: keep track of blocks that have no children (simplified)
        # For this prototype, we just link to the last N blocks added.
        self.tips: List[str] = [] 

    def add_block(self, data, validator_id) -> Block:
        """
        Creates and adds a new block to the DAG.
        Links to current 'tips' as parents.
        """
        # parents are the current tips. If genesis, empty list.
        parents = list(self.tips)
        
        new_block = Block(data, validator_id, parents)
        self.blocks[new_block.id] = new_block
        
        # In a real DAG, tips update is complex. 
        # Here: The new block becomes the new tip. 
        # Optionally keep some old tips if we want width > 1.
        # For simplicity: New block replaces all tips (linear-ish DAG) or appends.
        # Let's keep it linear-ish for simplicity unless we want to simulate forks.
        self.tips = [new_block.id]
        
        # print(f"DEBUG: Block {new_block.id} added by {validator_id}")
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

