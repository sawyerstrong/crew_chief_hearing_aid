"""crew_chief_hearing_aid — local voice control for CrewChief V4."""

import os as _os

# Must be set before pygame is imported anywhere, including by a bare
# __import__ in a dependency check. pygame prints a support banner to stdout on
# import, which corrupts the output of `bindings` and `doctor`.
_os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__version__ = "0.1.0"
