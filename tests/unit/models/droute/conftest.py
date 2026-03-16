try:
    import droute  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
