"""Connected components over accepted pairs, by union-find.

This is the only thing the GPL graph library was needed for, and it is fifteen lines. Keeping it
here means a caller gets clusters without inheriting a copyleft closure to compute them.

One caution worth stating: transitive closure is not always what a reviewer wants. If A matches B
and B matches C, this puts all three in one cluster even when A and C were never compared and
might disagree. `suspicious` reports clusters where that happened, so a person can look.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def clusters(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Members grouped by connectivity, each group sorted, the groups sorted."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    grouped: dict[str, list[str]] = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(members) for members in grouped.values()), key=lambda members: members[0])


def suspicious(pairs: Sequence[tuple[str, str]]) -> list[list[str]]:
    """Clusters held together by transitivity rather than by direct evidence.

    A cluster of n members supported by fewer than n-1 direct pairs cannot exist; one supported by
    exactly n-1 is a chain, where the ends were never compared. Those are the ones worth a look
    before anything is merged.
    """
    edges = {frozenset(pair) for pair in pairs if pair[0] != pair[1]}
    out: list[list[str]] = []
    for members in clusters(pairs):
        if len(members) < 3:
            continue
        inside = sum(1 for edge in edges if edge <= set(members))
        if inside <= len(members) - 1:
            out.append(members)
    return out


__all__ = ["clusters", "suspicious"]
