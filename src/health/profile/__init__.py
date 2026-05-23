"""Health Profile package (Phase G).

Phase G.1.a — crypto substrate (landed).
Phase G.1.b — schema + storage (landed this commit).
Phase G.1.c — CLI surface (next).
Phase G.1.d — REPL/Telegram/Signal integration + setup flow.
Phase G.2+   — extractor, slice selector, etc.
"""

from src.health.profile.crypto import (
    CryptoError,
    KeySource,
    decrypt_for_user,
    encrypt_for_user,
)
from src.health.profile.models import (
    Allergy,
    Concern,
    Condition,
    DeletedEntry,
    Demographics,
    FamilyHistoryEntry,
    HealthProfile,
    LabResult,
    LifestyleFactors,
    Medication,
    PendingUpdate,
    ProfileField,
)
from src.health.profile.store import (
    DEFAULT_USER_ID,
    LoadedProfile,
    StaleProfileError,
    init_tables,
    list_change_log,
    load_profile,
    save_profile,
)

__all__ = [
    "Allergy",
    "Concern",
    "Condition",
    "CryptoError",
    "DEFAULT_USER_ID",
    "DeletedEntry",
    "Demographics",
    "FamilyHistoryEntry",
    "HealthProfile",
    "KeySource",
    "LabResult",
    "LifestyleFactors",
    "LoadedProfile",
    "Medication",
    "PendingUpdate",
    "ProfileField",
    "StaleProfileError",
    "decrypt_for_user",
    "encrypt_for_user",
    "init_tables",
    "list_change_log",
    "load_profile",
    "save_profile",
]
