"""Backbone policy wrappers (ACT, PI0, PI05, stub)."""

from faact.backbone.base import BackboneFeatures

__all__ = [
    "Pi05PolicyWrapper",
    "PI0PolicyWrapper",
    "ACTPolicyWrapper",
    "BackboneFeatures",
    "StubBackboneWrapper",
]


def __getattr__(name: str):
    if name == "Pi05PolicyWrapper":
        from faact.backbone.pi05_wrapper import Pi05PolicyWrapper
        return Pi05PolicyWrapper
    if name == "PI0PolicyWrapper":
        from faact.backbone.pi0_wrapper import PI0PolicyWrapper
        return PI0PolicyWrapper
    if name == "ACTPolicyWrapper":
        from faact.backbone.act_wrapper import ACTPolicyWrapper
        return ACTPolicyWrapper
    if name == "StubBackboneWrapper":
        from faact.backbone.stub import StubBackboneWrapper
        return StubBackboneWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
