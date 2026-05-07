# Step 4+ plotting backends
__all__ = ["MatplotlibPlotter", "PlotlyPlotter"]

_LAZY_IMPORTS = {
    "MatplotlibPlotter": "src.plotting.backends.matplotlib_backend",
    "PlotlyPlotter": "src.plotting.backends.plotly_backend",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
