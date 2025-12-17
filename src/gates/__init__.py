"""
src.gates

Gate definitions and placeholders.

Defines placeholder gates used to represent distributed operations.

Public API:
- RemoteGatePlaceholder: Placeholder for cross-QPU gates.
- TeleportPlaceholder: Placeholder for qubit teleportation.
"""

from .placeholders import TeleportPlaceholder, RemoteGatePlaceholder

__all__ = ["TeleportPlaceholder", "RemoteGatePlaceholder"]
