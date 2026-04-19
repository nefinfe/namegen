"""namegen: research-grade realistic name generator.

Public API surface is intentionally small. See :mod:`namegen.api` for the real
types. Everything else is internal and subject to change between 0.x releases.
"""

from __future__ import annotations

from namegen.api import GeneratedName, Generator, Profile, ProfileError

__all__ = [
    "GeneratedName",
    "Generator",
    "Profile",
    "ProfileError",
    "__version__",
]

__version__ = "0.1.0"
