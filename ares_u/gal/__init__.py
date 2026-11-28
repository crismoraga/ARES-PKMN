"""ARES-U Game Abstraction Layer: one pixel/button interface for every mainline Pokemon console.

* ``games``    registry of the supported games (ROM, console, generation, metric adapters).
* ``backends`` emulator bridges with the ``BridgeLike`` protocol of ``ares_u.nav.env``:
  mGBA multi-core (GB, GBC, GBA) and DeSmuME (NDS).

The policy only ever sees pixels and presses the 8 universal buttons; RAM is read by the
adapters for *measurement* only (unique map/x/y cells), never for the observation.
"""
