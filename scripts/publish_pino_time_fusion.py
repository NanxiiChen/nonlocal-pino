"""Select reviewed temporal-control PDF/PNG pairs for local README/site use."""
import argparse
import json
from pathlib import Path

from scripts.publish_pino_three_method import publish


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--visual-review", required=True)
    parser.add_argument("--replace-provenance", type=Path,
                        help="Exact previously selected provenance; errors and weights must match")
    arguments = parser.parse_args()
    report = json.loads(arguments.provenance.read_text())
    if report["schema"] != "pino_time_fusion_figure_v1":
        raise ValueError("This entry requires audited time-fusion figures")
    print(json.dumps(publish(arguments.provenance, arguments.visual_review,
                             replace_provenance=arguments.replace_provenance), indent=2))
