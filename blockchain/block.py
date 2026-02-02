"""
Block Structure for Mocked DAG.

A block contains:
- ID
- Timestamp
- Data (Trust Updates)
- List of Parent Block IDs (DAG structure)
- Validator ID (who created it)
"""
import time
import uuid

class Block:
    def __init__(self, data, validator_id, parents):
        self.id = str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        self.data = data # In this project, this is the list of Trust Scores specific to an epoch
        self.validator_id = validator_id
        self.parents = parents # List of parent Block IDs
        
    def __repr__(self):
        return f"[Block {self.id} | Val: {self.validator_id} | Parents: {len(self.parents)}]"
