"""EntityKey service — KB §4, Milestone 7.

An entity's key is an **address**, never a feature (KB Decision 9). It selects
pixels and names a memory slot; it is never fed to the network as a number. The
engine already mints the key as ``"<session_id>:<instance_id>:<generation>"``,
and the ``generation`` field is what stops a recycled ``instance_id`` from
inheriting a destroyed entity's memory — M8's "destroyed entity cannot leak
memory to a later object" rests entirely on it, so it is parsed and enforced
here rather than assumed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKGROUND_ID = 0


@dataclass(frozen=True, order=True)
class EntityKey:
    session_id: str
    instance_id: int
    generation: int

    @classmethod
    def parse(cls, raw: str) -> "EntityKey":
        # rsplit, not split: session ids legitimately contain no ':' today, but a
        # key is "<session>:<instance>:<generation>" and only the LAST two fields
        # are fixed. Splitting from the left would break the moment a session id
        # ever gained a colon, and that failure would be silent.
        head, _, gen = raw.rpartition(":")
        session, _, inst = head.rpartition(":")
        if not session or not inst or not gen:
            raise ValueError(f"malformed entity key {raw!r}")
        return cls(session, int(inst), int(gen))

    def __str__(self) -> str:
        return f"{self.session_id}:{self.instance_id}:{self.generation}"

    @property
    def memory_slot(self) -> str:
        """What EPAM keys on (M8). Includes generation, so a recycled
        instance_id addresses a DIFFERENT slot than the entity it replaced."""
        return str(self)


def keys_for_frame(entities: list[dict]) -> dict[int, EntityKey]:
    """instance_id -> EntityKey for one frame's manifest entity records."""
    out: dict[int, EntityKey] = {}
    for e in entities:
        if e.get("destroyed"):
            continue
        iid = int(e["instance_id"])
        if iid == BACKGROUND_ID:
            continue
        out[iid] = EntityKey.parse(e["entity_key"])
    return out
