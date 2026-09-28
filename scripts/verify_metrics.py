"""Compare a fresh run against the committed metrics file.

Run in CI after `gatecheck run` to catch the case where a dependency bump or an
innocent looking refactor quietly moves a number the README quotes. Exits
non-zero and prints every drift it finds.

    python scripts/verify_metrics.py committed.json results/metrics.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# The environment block records versions and is expected to differ. Model derived
# scores get a looser tolerance because a solver can land on a slightly different
# optimum between library patch releases; everything else is seeded arithmetic and
# should reproduce far more tightly than this.
# ``figures`` is absent from a --skip-figures run, and ``environment`` records
# versions that are expected to differ between machines.
IGNORED_TOP_LEVEL = {"environment", "figures"}
LOOSE_KEYS = {"auc_with_engagement", "auc_day_one_only", "leak_contribution_auc"}
LOOSE_TOLERANCE = 1e-3
TIGHT_TOLERANCE = 1e-9


def walk(expected, actual, path: str, problems: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f"{path}: expected an object, found {type(actual).__name__}")
            return
        for key in sorted(set(expected) | set(actual)):
            if not path and key in IGNORED_TOP_LEVEL:
                continue
            if key not in expected:
                problems.append(
                    f"{path}/{key}: present in the new run, absent from the committed file"
                )
            elif key not in actual:
                problems.append(f"{path}/{key}: missing from the new run")
            else:
                walk(expected[key], actual[key], f"{path}/{key}", problems)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            problems.append(f"{path}: list length changed")
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            walk(left, right, f"{path}[{index}]", problems)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            problems.append(f"{path}: {expected} became {actual}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        tolerance = LOOSE_TOLERANCE if path.rsplit("/", 1)[-1] in LOOSE_KEYS else TIGHT_TOLERANCE
        if not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance):
            problems.append(f"{path}: {expected!r} became {actual!r}")
        return
    if expected != actual:
        problems.append(f"{path}: {expected!r} became {actual!r}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    fresh = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    problems: list[str] = []
    walk(committed, fresh, "", problems)
    if problems:
        print(f"{len(problems)} value(s) drifted from the committed run:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("every committed number reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
