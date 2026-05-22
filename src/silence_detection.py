"""Backward-compatible re-export of quiet-detection symbols.

Prefer importing from :mod:`contact_detection` or :mod:`contact_detection.quiet`.
"""

try:
    from .contact_detection.quiet import *  # noqa: F403
except ImportError:  # pragma: no cover - supports PYTHONPATH=src imports
    from contact_detection.quiet import *  # noqa: F403
