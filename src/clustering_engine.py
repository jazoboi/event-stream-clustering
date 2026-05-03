"""
K-Means clustering engine with periodic online updates.

Maintains a clustering model that refits on accumulated
micro-batches for adaptive event categorization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    """Cluster assignment for a single event."""
    cluster_id: int
    priority_label: str
    distance_to_centroid: float


PRIORITY_MAP = {
    0: "Critical",
    1: "High",
    2: "Standard",
    3: "Maintenance",
    4: "Informational",
}


class StreamingClusterEngine:
    """Online K-Means clustering for event streams.

    Uses MiniBatchKMeans for incremental updates as new
    micro-batches arrive from the streaming pipeline.

    Parameters
    ----------
    n_clusters : int
        Number of event priority tiers (default: 5).
    refit_interval : int
        Number of batches between full refits (default: 100).
    """

    def __init__(self, n_clusters: int = 5, refit_interval: int = 100) -> None:
        self._n_clusters = n_clusters
        self._refit_interval = refit_interval
        self._model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._batch_count = 0
        self._fitted = False

    def partial_fit(self, features: np.ndarray) -> float:
        """Incrementally update the clustering model.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix from the latest micro-batch.

        Returns
        -------
        float
            Current silhouette score (or -1 if not enough data).
        """
        if not self._fitted:
            self._scaler.fit(features)
            self._fitted = True

        scaled = self._scaler.transform(features)
        self._model.partial_fit(scaled)
        self._batch_count += 1

        if len(scaled) > self._n_clusters:
            labels = self._model.predict(scaled)
            return float(silhouette_score(scaled, labels))
        return -1.0

    def predict(self, features: np.ndarray) -> list[ClusterAssignment]:
        """Assign cluster labels to new events.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix for events to classify.

        Returns
        -------
        list[ClusterAssignment]
            Cluster assignments with priority labels.
        """
        scaled = self._scaler.transform(features)
        labels = self._model.predict(scaled)
        distances = self._model.transform(scaled)

        assignments = []
        for i, label in enumerate(labels):
            assignments.append(ClusterAssignment(
                cluster_id=int(label),
                priority_label=PRIORITY_MAP.get(int(label), "Unknown"),
                distance_to_centroid=float(distances[i, label]),
            ))
        return assignments
