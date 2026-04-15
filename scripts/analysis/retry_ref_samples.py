"""
Retry failed ref_samples FSCS cells by re-running sample_reference_points.py
for each ecoregion that has unprocessed cells.

This simply calls the existing extraction script which supports checkpoint
resume — it loads existing data, skips processed cells, and only attempts
the failed ones.

Usage:
    python scripts/analysis/retry_ref_samples.py
    python scripts/analysis/retry_ref_samples.py --eco_id 88
"""
import sys
import json
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DATA = Path(r"c:\Users\coach\myfiles\postdoc2\code\data")
PYTHON = r"C:\Users\coach\.conda\envs\erthy\python.exe"
SCRIPT = r"c:\Users\coach\myfiles\postdoc2\code\scripts\extraction\sample_reference_points.py"

ECOS_WITH_CP = [40, 89, 90, 110, 16, 102, 94, 15, 65,
                41, 38, 97, 48, 101, 88]


def get_failed_cells(eco_id):
    cp = DATA / f"ref_samples_eco{eco_id}.checkpoint.json"
    if not cp.exists():
        return []
    with open(cp) as f:
        processed = set(json.load(f))
    max_idx = max(processed) if processed else 0
    return sorted(set(range(max_idx + 1)) - processed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eco_id", type=int, default=None)
    args = parser.parse_args()

    ecos = [args.eco_id] if args.eco_id else ECOS_WITH_CP

    for eco_id in ecos:
        failed = get_failed_cells(eco_id)
        if not failed:
            print(f"eco{eco_id}: All cells done")
            continue

        print(f"\n{'='*60}")
        print(f"eco{eco_id}: Retrying {len(failed)} failed cells: "
              f"{failed[:10]}{'...' if len(failed) > 10 else ''}")
        print(f"{'='*60}")

        result = subprocess.run(
            [PYTHON, SCRIPT, "--eco_id", str(eco_id)],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] eco{eco_id} exited with code {result.returncode}")

        # Check remaining failures
        remaining = get_failed_cells(eco_id)
        if remaining:
            print(f"  eco{eco_id}: Still {len(remaining)} cells failed: "
                  f"{remaining[:10]}")
        else:
            print(f"  eco{eco_id}: All cells now processed!")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL REF_SAMPLES STATUS")
    print(f"{'='*60}")
    total_fail = 0
    for eco_id in ECOS_WITH_CP:
        failed = get_failed_cells(eco_id)
        if failed:
            print(f"  eco{eco_id}: {len(failed)} cells still failed")
            total_fail += len(failed)
    if total_fail == 0:
        print("  All cells processed across all ecoregions!")
    else:
        print(f"\n  Total remaining failures: {total_fail}")


if __name__ == "__main__":
    main()
