try:
    import jsnow17  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
