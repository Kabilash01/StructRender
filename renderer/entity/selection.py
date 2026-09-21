"""Entity token overflow policy — KB §4, Milestone 7 exit criterion 3.

When a frame has more visible entities than ``K_max`` slots, which ones get a
token is decided by **visibility rank**: descending ``pixel_count``, never spawn
order, never dictionary order, never whatever the engine happened to emit first.
This is a locked decision (CLAUDE.md: "K_max entity-token overflow is resolved by
ranking visible entities by pixel_count, never by arbitrary/spawn order") and the
milestone's exit criterion is that overflow is **deterministic**.

Ties are the part that makes "deterministic" non-trivial: two entities with the
same pixel_count must still order the same way on every run and every machine, so
the tie-break is the entity key's ``(instance_id, generation)`` — engine-owned,
stable, and total. Sorting by pixel_count alone would leave ties to the input
order, which is exactly the "arbitrary order" the criterion forbids.
"""

from __future__ import annotations

import numpy as np

from .keys import BACKGROUND_ID, EntityKey


def pixel_counts(entity_id_map: np.ndarray, *, min_px: int = 1) -> dict[int, int]:
    """instance_id -> visible pixel count, background excluded."""
    ids, counts = np.unique(entity_id_map, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)
            if int(i) != BACKGROUND_ID and int(c) >= min_px}


def select_entities(counts: dict[int, int], keys: dict[int, EntityKey],
                    k_max: int) -> list[int]:
    """The <= k_max instance_ids that get a token, in slot order.

    Descending pixel_count; ties broken by (instance_id, generation) so the order
    is total and reproducible. Entities with no key are dropped rather than given
    a slot — an entity the engine did not describe has no address, and inventing
    one would put an unidentified object in a memory slot.
    """
    ranked = sorted(
        (iid for iid in counts if iid in keys),
        key=lambda i: (-counts[i], keys[i].instance_id, keys[i].generation),
    )
    return ranked[:k_max]


def build_entity_masks(entity_id_map: np.ndarray, selected: list[int],
                       k_max: int) -> np.ndarray:
    """[K_max, H, W] float32 masks; unused slots are all-zero.

    All-zero padded slots are load-bearing, not filler: the attention module's
    tests assert a padded slot cannot contribute to any pixel, so the padding IS
    the mechanism that stops an empty slot leaking into the output.
    """
    out = np.zeros((k_max, *entity_id_map.shape), dtype=np.float32)
    for slot, iid in enumerate(selected[:k_max]):
        out[slot] = (entity_id_map == iid).astype(np.float32)
    return out
