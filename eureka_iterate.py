"""
Eureka multi-variant harness.

Runs K cost-function variants concurrently, M games each, aggregates
win rate and per-component cost statistics per variant, prints a
side-by-side comparison table.

Each game is its own subprocess running scenario_circle.py with the
COST_MODULE env var set to a different cost variant module.

Usage:
    python eureka_iterate.py cost_circle cost_circle_v1a cost_circle_v1b cost_circle_v1c
    GAMES_PER_VARIANT=8 python eureka_iterate.py cost_circle cost_circle_v1a
"""

import os
import re
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List

import numpy as np


DEFAULT_SCENARIO = "scenario_circle.py"
SC2PATH_DEFAULT = "/home/pyesley/StarCraftII"
DEFAULT_GAMES_PER_VARIANT = 5


def run_one(variant: str, game_id: int, scenario: str, sc2path: str, timeout: int) -> dict:
    """Run one game; subprocess uses the chosen cost variant via env."""
    env = os.environ.copy()
    env["SC2PATH"] = sc2path
    env["COST_MODULE"] = variant
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, scenario],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    wall = time.time() - t0
    out = proc.stdout + "\n" + proc.stderr

    result = "ERROR"
    reason = "no GAME OVER line"
    m = re.search(r"GAME OVER:\s+(\w+)\s+—\s+(.*)", out)
    if m:
        result = m.group(1)
        reason = m.group(2).strip()

    game_time = None
    steps = None
    gt = re.search(r"Game time:\s+([\d.]+)s", out)
    if gt:
        game_time = float(gt.group(1))
    st = re.search(r"Steps:\s+(\d+)", out)
    if st:
        steps = int(st.group(1))

    components: Dict[str, float] = {}
    for line in out.splitlines():
        cm = re.match(r"^\s+(\w+)\s+:\s+([-\d.]+)\s*$", line)
        if cm:
            components[cm.group(1)] = float(cm.group(2))

    return {
        "variant": variant,
        "id": game_id,
        "result": result,
        "reason": reason,
        "game_time": game_time,
        "steps": steps,
        "wall_time": wall,
        "components": components,
    }


def comparison_table(by_variant: Dict[str, List[dict]], variants: List[str]) -> None:
    """Print side-by-side variant stats."""
    # Outcome counts + win rate
    print()
    print("=" * 90)
    print("EUREKA COMPARISON")
    print("=" * 90)
    header = f"  {'variant':25s} {'W':>3} {'L':>3} {'TO':>3} {'ER':>3} {'win%':>5} " \
             f"{'avg game (s)':>13} {'avg wall (s)':>13}"
    print(header)
    print("  " + "-" * 87)
    for v in variants:
        rs = by_variant.get(v, [])
        if not rs:
            print(f"  {v:25s}   no results")
            continue
        wins = sum(1 for r in rs if r["result"] == "WIN")
        losses = sum(1 for r in rs if r["result"] == "LOSS")
        timeouts = sum(1 for r in rs if r["result"] == "TIMEOUT")
        errors = sum(1 for r in rs if r["result"] not in ("WIN", "LOSS", "TIMEOUT"))
        n = len(rs)
        wr = 100.0 * wins / n
        gts = [r["game_time"] for r in rs if r["game_time"] is not None]
        wts = [r["wall_time"] for r in rs]
        avg_gt = np.mean(gts) if gts else float("nan")
        avg_wt = np.mean(wts) if wts else float("nan")
        print(f"  {v:25s} {wins:>3} {losses:>3} {timeouts:>3} {errors:>3} "
              f"{wr:>4.0f}% {avg_gt:>11.1f}   {avg_wt:>11.1f}")

    # Per-component table
    all_keys = set()
    for v in variants:
        for r in by_variant.get(v, []):
            all_keys |= set(r["components"].keys())
    if not all_keys:
        return

    print(f"\n  Per-component cost (mean across games per variant):")
    short_names = [v.replace("cost_circle", "").lstrip("_") or "BASELINE" for v in variants]
    head = f"  {'component':25s}" + "".join(f" {sn[:10]:>13s}" for sn in short_names)
    print(head)
    print("  " + "-" * (25 + 14 * len(variants)))
    for k in sorted(all_keys):
        row = f"  {k:25s}"
        for v in variants:
            vals = [r["components"].get(k, 0.0) for r in by_variant.get(v, []) if r["components"]]
            avg = np.mean(vals) if vals else float("nan")
            row += f" {avg:>+12.0f} "
        print(row)
    print("=" * 90)

    # Per-game one-liners grouped by variant
    print("\nPer-game outcomes:")
    for v in variants:
        rs = sorted(by_variant.get(v, []), key=lambda x: x["id"])
        if not rs:
            continue
        print(f"  [{v}]")
        for r in rs:
            gt = f"{r['game_time']:5.1f}s" if r["game_time"] else "  ---"
            print(f"     game {r['id']:2d}: {r['result']:8s} game={gt}  reason: {r['reason']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    variants = sys.argv[1:]
    games = int(os.environ.get("GAMES_PER_VARIANT", str(DEFAULT_GAMES_PER_VARIANT)))
    timeout = int(os.environ.get("RUN_TIMEOUT", "600"))
    sc2path = os.environ.get("SC2PATH", SC2PATH_DEFAULT)
    scenario = os.environ.get("SCENARIO", DEFAULT_SCENARIO)
    total = len(variants) * games

    print(f"Eureka iteration: {len(variants)} variants × {games} games = {total} games")
    print(f"  Scenario : {scenario}")
    print(f"  Variants : {', '.join(variants)}")
    print(f"  SC2PATH  : {sc2path}")
    print(f"  Timeout  : {timeout}s/game")
    print()

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=total) as ex:
        futures = {}
        for v in variants:
            for g in range(games):
                fut = ex.submit(run_one, v, g, scenario, sc2path, timeout)
                futures[fut] = (v, g)
        for fut in as_completed(futures):
            v, g = futures[fut]
            try:
                r = fut.result()
            except subprocess.TimeoutExpired:
                r = {"variant": v, "id": g, "result": "ERROR",
                     "reason": "subprocess timeout", "game_time": None,
                     "steps": None, "wall_time": timeout, "components": {}}
            except Exception as e:
                r = {"variant": v, "id": g, "result": "ERROR",
                     "reason": f"exception: {e}", "game_time": None,
                     "steps": None, "wall_time": -1, "components": {}}
            results.append(r)
            tag = v.replace("cost_circle", "").lstrip("_") or "BASE"
            print(f"  [{tag:8s} g{g}] {r['result']:8s} "
                  f"game={r['game_time']} wall={r['wall_time']:.1f}s")

    print(f"\nTotal wall: {time.time() - t0:.1f}s")

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)
    comparison_table(by_variant, variants)


if __name__ == "__main__":
    main()
