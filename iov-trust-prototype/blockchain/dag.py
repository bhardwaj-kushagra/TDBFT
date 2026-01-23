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
