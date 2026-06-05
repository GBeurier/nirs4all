"""Contract snapshot tests for the stable 0.9.x public surface.

These tests freeze the *current* public contract so that future refactors
cannot silently drift it. They are deliberately additive and introspective:
they read the live library and compare it against hard-coded snapshots that
were captured from the current code. A diff here means a stable contract
(public signatures, ``__all__``, on-disk SQLite schema, bundle format) changed
and the change must be intentional (and, per the 0.9.x policy, gated behind a
major bump).
"""
