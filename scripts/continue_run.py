import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from compact_medvqa.pipeline import main

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Continue/replay a cached Compact MedVQA v5 run"
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--fast-subset", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-reports", action="store_true")
    p.add_argument("--skip-qualitative", action="store_true")
    args = p.parse_args()
    main(
        mode="continue",
        datasets=args.datasets,
        fast_subset=args.fast_subset if args.fast_subset else None,
        skip_ablations=args.skip_ablations,
        skip_robustness=args.skip_robustness,
        skip_figures=args.skip_figures,
        skip_reports=args.skip_reports,
        skip_qualitative=args.skip_qualitative,
        skip_warmup=True,
    )
