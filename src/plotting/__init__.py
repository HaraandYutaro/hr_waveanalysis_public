# Step 4+ plotting package
__all__ = ["PlotterBase", "PlotterWrapperMixin"]

_LAZY_IMPORTS = {
    "PlotterBase": "src.plotting.backend_base",
    "PlotterWrapperMixin": "src.plotting.wrapper",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
