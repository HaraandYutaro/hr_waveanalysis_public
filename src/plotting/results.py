from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResult:
    backend_name: str
    plot_type: str
    figure: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatplotlibBackendResult(BackendResult):
    axes: list[Any] = field(default_factory=list)
    artists: list[Any] = field(default_factory=list)
    colorbars: list[Any] = field(default_factory=list)


@dataclass
class PlotlyBackendResult(BackendResult):
    traces: list[Any] = field(default_factory=list)
    layout: Any | None = None