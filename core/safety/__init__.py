"""Integer battle damage and conditional survival proofs for generation three."""

from core.safety.z3_integer_verifier import (
    DamageContext,
    DamageResult,
    GbaIntegerSafetyVerifier,
    ResidualContext,
    compute_residual_damage,
)

__all__ = [
    "DamageContext", "DamageResult", "GbaIntegerSafetyVerifier", "ResidualContext",
    "compute_residual_damage",
]
