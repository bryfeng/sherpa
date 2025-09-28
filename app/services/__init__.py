"""Service layer helpers"""

from .entitlement import EntitlementError, evaluate_entitlement

__all__ = ["evaluate_entitlement", "EntitlementError"]
