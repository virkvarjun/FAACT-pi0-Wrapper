"""Backbone policy wrappers (π₀.5 inference)."""

from faact.backbone.base import BackboneFeatures

__all__ = ["Pi05PolicyWrapper", "BackboneFeatures", "StubBackboneWrapper"]


def __getattr__(name: str):
    if name == "Pi05PolicyWrapper":
        from faact.backbone.pi05_wrapper import Pi05PolicyWrapper
        return Pi05PolicyWrapper
    if name == "StubBackboneWrapper":
        from faact.backbone.stub import StubBackboneWrapper
        return StubBackboneWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
