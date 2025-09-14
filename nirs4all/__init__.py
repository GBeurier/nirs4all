try:
    from importlib.metadata import version
    __version__ = version("nirs4all")
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("nirs4all").version
    except Exception:
        __version__ = "unknown"

__all__ = ["core", "data", "data_splitters", "presets", "transformations", "utils", "cli"]
