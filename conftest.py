import sys
from pathlib import Path

# Add src/ to the path so tests can import bandits, features, etc. directly.
sys.path.insert(0, str(Path(__file__).parent / "src"))