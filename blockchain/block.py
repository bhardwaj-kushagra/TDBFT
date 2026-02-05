"""
Block Structure for Mocked DAG.

A block contains:
- ID
- Timestamp
- Data (Trust Updates)
- List of Parent Block IDs (DAG structure)
- Validator ID (who created it)
- Issuer Trust (at time of creation)
- TCW (Trust Weighted Cumulative Weight)
"""
import time
import uuid

class Block:
    def __init__(self, data, validator_id, parents, issuer_trust=0.0, step=0):
        self.id = str(uuid.uuid4())[:8]
        # Use logical simulation step instead of wall-clock time for reproducibility
        self.timestamp = step 
        self.data = data # In this project, this is the list of Trust Scores specific to an epoch
        self.validator_id = validator_id
        self.parents = parents # List of parent Block IDs
        self.issuer_trust = issuer_trust
        self.tcw = issuer_trust # Initialize with own trust (Step 11 requirement)
        
    def __repr__(self):
        return f"[Block {self.id} | Val: {self.validator_id} | Trust: {self.issuer_trust:.2f} | TCW: {self.tcw:.2f}]"
