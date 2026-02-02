"""
Base Trust Strategy.

Defines the interface that all trust models must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

class TrustStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def record_interaction(self, vehicle: Any, target_id: str, is_positive: bool):
        """
        Updates local trust state based on a new interaction.
        Args:
            vehicle: The Vehicle instance recording the interaction.
            target_id: The ID of the target vehicle.
            is_positive: Result of the interaction.
        """
        pass

    @abstractmethod
    def get_local_trust(self, vehicle: Any, target_id: str) -> float:
        """
        Computes the local trust score for a target.
        """
        pass

    @abstractmethod
    def get_trust_reports(self, vehicle: Any) -> Dict:
        """
        Returns the raw data needed by RSU for aggregation.
        """
        pass

    @abstractmethod
    def compute_global_trust(self, rsu: Any, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        """
        Aggregates local reports into global trust scores.
        Args:
            rsu: The RSU instance performing aggregation.
            all_ids: List of all vehicle IDs.
            reports: Dictionary of reports from all vehicles.
        Returns:
            Dict mapping vehicle_id to global trust score [0.0, 1.0].
        """
        pass
