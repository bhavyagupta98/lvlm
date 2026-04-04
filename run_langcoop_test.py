#!/usr/bin/env python3
"""Compatibility wrapper for the single-ego LangCoop runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_runner.runners.single import main


if __name__ == "__main__":
    main()
