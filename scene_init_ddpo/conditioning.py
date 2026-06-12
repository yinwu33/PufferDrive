"""Conditioning pool: real maps used to condition scene generation.

The DiT needs a full heterogeneous conditioning graph per scene (``condition``,
``lg_type``, ``map_id``, lane/agent nodes, the three edge sets, ...). Rather than
re-vendor scenario-dreamer's entire dataset/datamodule, we **dump** a pool of
already-built ``HeteroData`` graphs offline in the scenario-dreamer venv (see
``tools/dump_conditioning.py``) and here simply load and batch them. Both venvs
pin torch_geometric 2.6.1, so the pickled graphs are portable.

Each conditioning graph must already carry everything ``DiT.forward`` reads:
    data['condition'], data['lg_type'], data['map_id'],
    data['num_agents'], data['num_lanes'],
    data['agent'].x / .type, data['lane'].x,
    ('lane','to','lane').edge_index, ('agent','to','agent').edge_index,
    ('lane','to','agent').edge_index
Only the map (lane geometry + lane edges) and the per-scene agent COUNT are used;
the real agent state values (``agent.x``) are ignored — every agent, including the
ego and its goal, is generated from noise (see ``scene_models/dm_goal.py``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch_geometric.data import Batch

# The conditioning graphs are ``ScenarioDreamerData`` instances pickled in the
# scenario-dreamer venv under the module path ``utils.data_container``. That class
# is vendored here identically (sd_model/data_container.py); alias the original
# module path into sys.modules so torch.load can resolve the pickled class without
# scenario-dreamer on the path.
from .sd_model import data_container as _sd_data_container

if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")
sys.modules.setdefault("utils.data_container", _sd_data_container)


class ConditioningPool:
    """Loads a pool of conditioning ``HeteroData`` graphs and serves random batches."""

    def __init__(self, pool_path: str | Path, device: str = "cuda", seed: int = 0):
        pool_path = Path(pool_path)
        if pool_path.is_dir():
            self.graphs = [torch.load(p, weights_only=False) for p in sorted(pool_path.glob("*.pt"))]
        else:
            self.graphs = torch.load(pool_path, weights_only=False)
        if not self.graphs:
            raise FileNotFoundError(f"No conditioning graphs found at {pool_path}")
        self.device = device
        self.g = torch.Generator().manual_seed(seed)
        print(f"[conditioning] loaded {len(self.graphs)} conditioning graphs")

    def __len__(self) -> int:
        return len(self.graphs)

    def sample_batch(self, batch_size: int) -> Batch:
        """Random batch of conditioning scenes (with replacement)."""
        idx = torch.randint(0, len(self.graphs), (batch_size,), generator=self.g).tolist()
        return self._collate([self.graphs[i] for i in idx])

    def batch_from_indices(self, indices) -> Batch:
        """Deterministic batch of specific pool entries (for stable eval visuals)."""
        return self._collate([self.graphs[i] for i in indices])

    def iter_epoch(self, batch_size: int):
        """Iterate the pool once in fixed-size batches (drops the remainder)."""
        order = torch.randperm(len(self.graphs), generator=self.g).tolist()
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield self._collate([self.graphs[i] for i in order[start : start + batch_size]])

    def _collate(self, graphs) -> Batch:
        batch = Batch.from_data_list(graphs)
        # PyG sets batch_size from the number of graphs; make sure it is present.
        batch.batch_size = len(graphs)
        return batch.to(self.device)
