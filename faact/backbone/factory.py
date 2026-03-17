"""Factory helpers for FAACT backbone wrappers."""

from __future__ import annotations

from typing import Any

from faact.backbone import ACTPolicyWrapper, PI0PolicyWrapper, Pi05PolicyWrapper, StubBackboneWrapper
from faact.backbone.base import BackbonePolicyWrapper


def make_backbone_wrapper(
    policy_type: str,
    checkpoint_path: str,
    device: str = "cuda",
    task_desc: str | None = None,
    **kwargs: Any,
) -> BackbonePolicyWrapper:
    """Instantiate a canonical wrapper for the requested backbone."""
    policy_type = policy_type.lower()
    if policy_type == "act":
        return ACTPolicyWrapper(checkpoint_path, device=device, **kwargs)
    if policy_type == "pi0":
        return PI0PolicyWrapper(checkpoint_path, device=device, task_default=task_desc, **kwargs)
    if policy_type == "pi05":
        return Pi05PolicyWrapper(checkpoint_path, device=device, task_default=task_desc or "pick up the object", **kwargs)
    if policy_type == "stub":
        return StubBackboneWrapper(**kwargs)
    raise ValueError(f"Unsupported policy_type: {policy_type}")
