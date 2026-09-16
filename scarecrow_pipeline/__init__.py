"""Data and image-analysis primitives for the Scarecrow VN pipeline.

Submodule imports (e.g. `scarecrow_pipeline.environment_asset`) do not trigger
this file's __getattr__ and stay pydantic-free. Only attribute access on the
package itself (e.g. `scarecrow_pipeline.RenderRequest`) lazily imports
schemas.py, so code running under Blender's bundled Python (no pydantic) can
reach bpy-free leaf modules without a pydantic import ever occurring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import FACSExpression, PosePayload, RenderRequest, SceneLighting

__all__ = ["RenderRequest", "SceneLighting", "PosePayload", "FACSExpression"]

_SCHEMA_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name in _SCHEMA_EXPORTS:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
