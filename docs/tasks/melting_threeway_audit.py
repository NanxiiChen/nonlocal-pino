"""CPU-only saved-state/field audit entry for the finite melting comparison."""
import argparse
from pathlib import Path

from docs.tasks.melting_threeway_queue import audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    audit(args.bundle, args.prefix)
