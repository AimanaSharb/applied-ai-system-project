"""Put the project root on sys.path so `pytest` works from any invocation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
