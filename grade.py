"""Regenerate every corpus verdict from scratch. Kept for backward compatibility:

    python grade.py        (same as: trajlens corpus)
"""
import sys

from trajlens.cli import main

if __name__ == "__main__":
    sys.exit(main(["corpus", *sys.argv[1:]]))
