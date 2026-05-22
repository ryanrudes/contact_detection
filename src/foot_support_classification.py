"""Backward-compatible re-export of per-foot support classification symbols.

Prefer importing from :mod:`contact_detection` or :mod:`contact_detection.foot_support`.
"""

try:
    from .contact_detection.foot_support import *  # noqa: F403
except ImportError:  # pragma: no cover - supports PYTHONPATH=src imports
    from contact_detection.foot_support import *  # noqa: F403
