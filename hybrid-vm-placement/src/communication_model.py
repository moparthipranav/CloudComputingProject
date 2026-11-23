"""Sparse VM-to-VM communication model and synthetic traffic generator.

Traffic is stored as a symmetric CSR structure (``indptr``/``indices``) with three
aligned data arrays per directed entry: frequency (messages/s), volume (MB/s) and
intensity (normalised blend of the two, in [0, 1]). Each unordered VM pair appears
twice (once per row), so memory is O(pairs), never O(V^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

KINDS = ("frequency", "volume", "intensity")


@dataclass
class CommunicationModel:
    num_vms: int
    indptr: np.ndarray      # (V+1,) int64
    indices: np.ndarray     # (2E,) int32  neighbour VM ids
    frequency: np.ndarray   # (2E,) float64
    volume: np.ndarray      # (2E,) float64
    intensity: np.ndarray   # (2E,) float64
    group_ids: np.ndarray | None = None

    @property
    def num_pairs(self) -> int:
        return int(self.indices.size // 2)

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.indptr, self.indices, self.frequency,
                                          self.volume, self.intensity)))

    @classmethod
    def from_pairs(cls, n: int, src: np.ndarray, dst: np.ndarray, freq: np.ndarray,
                   vol: np.ndarray, intensity: np.ndarray | None = None,
                   group_ids: np.ndarray | None = None) -> "CommunicationModel":
        """Build the symmetric structure from unique unordered pairs (src != dst)."""
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        if src.size and (np.any(src == dst) or src.min() < 0 or dst.max() >= n or src.max() >= n):
            raise ValueError("Invalid pair list (self loops or out-of-range VM id)")
        freq = np.asarray(freq, dtype=np.float64)
        vol = np.asarray(vol, dtype=np.float64)
        if intensity is None:
            fmax = freq.max() if freq.size else 1.0
            vmax = vol.max() if vol.size else 1.0
            intensity = 0.5 * freq / fmax + 0.5 * vol / vmax
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])

        def sym(a: np.ndarray) -> np.ndarray:
            return np.concatenate([a, a])[order].astype(np.float64)

        return cls(n, indptr, cols.astype(np.int32), sym(freq), sym(vol), sym(intensity),
                   group_ids)

    def weights(self, kind: str) -> np.ndarray:
        if kind not in KINDS:
            raise ValueError(f"Unknown weight kind {kind!r}; expected one of {KINDS}")
        return getattr(self, kind)

    def csr(self, kind: str = "frequency") -> csr_matrix:
        return csr_matrix((self.weights(kind), self.indices, self.indptr),
                          shape=(self.num_vms, self.num_vms))

    def to_dense(self, kind: str = "intensity") -> np.ndarray:
        return self.csr(kind).toarray()

    def pair_arrays(self, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unordered pairs (src < dst) with the chosen weight."""
        rows = np.repeat(np.arange(self.num_vms), np.diff(self.indptr))
        mask = rows < self.indices
        return rows[mask], self.indices[mask].astype(np.int64), self.weights(kind)[mask]

    def neighbor_lists(self, kind: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-VM (neighbour ids, weights) array views for fast incremental updates."""
        w = self.weights(kind)
        nb = [self.indices[self.indptr[v]:self.indptr[v + 1]] for v in range(self.num_vms)]
        ws = [w[self.indptr[v]:self.indptr[v + 1]] for v in range(self.num_vms)]
        return nb, ws

    def strength(self, kind: str) -> np.ndarray:
        """Total traffic weight per VM."""
        rows = np.repeat(np.arange(self.num_vms), np.diff(self.indptr))
        return np.bincount(rows, weights=self.weights(kind), minlength=self.num_vms)


def _sample_unique_keys(rng: np.random.Generator, draw: Callable[[int], np.ndarray],
                        count: int, exclude: np.ndarray) -> np.ndarray:
    """Draw ``count`` unique pair keys not in ``exclude``."""
    acc = np.empty(0, dtype=np.int64)
    for _ in range(500):
        if acc.size >= count:
            break
        batch = np.unique(draw(int((count - acc.size) * 1.3) + 32))
        if exclude.size:
            batch = batch[~np.isin(batch, exclude)]
        if acc.size:
            batch = batch[~np.isin(batch, acc)]
        acc = np.concatenate([acc, batch])
    if acc.size < count:
        raise RuntimeError("Could not sample enough unique communication pairs")
    return rng.permutation(acc)[:count] if acc.size > count else acc


def generate_communication(
    num_vms: int, rng: np.random.Generator, num_relationships: int | None = None,
    relationships_per_vm: float = 5.0, group_size_range: tuple[int, int] = (4, 24),
    intra_group_fraction: float = 0.85, freq_median_intra: float = 40.0,
    freq_median_inter: float = 4.0, freq_sigma: float = 1.0,
    msg_kb_median: float = 64.0, msg_kb_sigma: float = 0.8,
) -> CommunicationModel:
    """Synthetic multi-tier application traffic.

    VMs are split into application groups; most pairs are intra-group with heavy,
    heavy-tailed (log-normal) traffic, the rest are light inter-group links.
    """
    n = num_vms
    if n < 2:
        raise ValueError("Need at least 2 VMs")
    max_pairs = n * (n - 1) // 2
    target = num_relationships if num_relationships is not None \
        else int(round(relationships_per_vm * n))
    target = int(min(max(target, 1), max_pairs))

    lo, hi = max(2, int(group_size_range[0])), max(2, int(group_size_range[1]))
    sizes: list[int] = []
    remaining = n
    while remaining > 0:
        s = min(int(rng.integers(lo, hi + 1)), remaining)
        sizes.append(s)
        remaining -= s
    if len(sizes) > 1 and sizes[-1] < 2:
        last = sizes.pop()
        sizes[-1] += last  # merge a trailing singleton into the previous group
    sz = np.array(sizes, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sz)[:-1]])
    order = rng.permutation(n)
    group_ids = np.empty(n, dtype=np.int64)
    group_ids[order] = np.repeat(np.arange(sz.size), sz)

    pairs_per_group = sz * (sz - 1) // 2
    n_intra = min(int(round(target * intra_group_fraction)), int(pairs_per_group.sum()))
    keys = np.empty(0, dtype=np.int64)
    if n_intra > 0:
        pw = pairs_per_group / pairs_per_group.sum()

        def draw_intra(b: int) -> np.ndarray:
            g = rng.choice(sz.size, size=b, p=pw)
            s = sz[g]
            a = (rng.random(b) * s).astype(np.int64)
            o = (rng.random(b) * (s - 1)).astype(np.int64)
            o += (o >= a)
            u, v = order[starts[g] + a], order[starts[g] + o]
            return np.minimum(u, v) * n + np.maximum(u, v)

        keys = _sample_unique_keys(rng, draw_intra, n_intra, np.empty(0, dtype=np.int64))

    n_inter = target - keys.size
    if n_inter > 0:
        def draw_any(b: int) -> np.ndarray:
            u = rng.integers(0, n, b)
            v = rng.integers(0, n - 1, b)
            v += (v >= u)
            return np.minimum(u, v) * n + np.maximum(u, v)

        keys = np.concatenate([keys, _sample_unique_keys(rng, draw_any, n_inter, keys)])

    keys = np.sort(keys)
    src, dst = keys // n, keys % n
    intra = group_ids[src] == group_ids[dst]
    median = np.where(intra, freq_median_intra, freq_median_inter)
    freq = np.clip(np.exp(rng.normal(np.log(median), freq_sigma)), 0.1, 5000.0)
    size_kb = np.exp(rng.normal(np.log(msg_kb_median), msg_kb_sigma))
    vol = freq * size_kb / 1024.0  # MB/s
    return CommunicationModel.from_pairs(n, src, dst, freq, vol, group_ids=group_ids)
