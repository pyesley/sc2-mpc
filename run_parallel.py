"""
Run N games of a scenario in parallel and aggregate stats.

Each game runs in its own subprocess (separate Python interpreter,
separate SC2 instance, independent RNG). Output is parsed for the
GAME OVER line and the per-component cost block.

Usage:
    python run_parallel.py                       # 4 games of scenario_circle
    python run_parallel.py 8                     # 8 games of scenario_circle
    python run_parallel.py 4 scenario_3v2.py     # 4 games of a different scenario
"""

import os
import re
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


SC2PATH_DEFAULT = "/home/pyesley/StarCraftII"


def run_one(game_id: int, script: str, sc2path: str, timeout: int) -> dict:
    """Run one scenario subprocess; parse GAME OVER and components."""
    env = os.environ.copy()
    env["SC2PATH"] = sc2path
    # Pin BLAS / numpy thread pools to 1 per game subprocess. Otherwise
    # OpenBLAS / MKL spin up N_CPU threads per process, and N_games × N_CPU
    # threads thrash. Single-threaded numpy is ~the same speed for our
    # batch sizes and lets games scale linearly with worker count.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.time() - t0
    out = proc.stdout + "\n" + proc.stderr

    result = "ERROR"
    reason = "no GAME OVER line"
    m = re.search(r"GAME OVER:\s+(\w+)\s+—\s+(.*)", out)
    if m:
        result = m.group(1)
        reason = m.group(2).strip()

    game_time: Optional[float] = None
    steps: Optional[int] = None
    gt = re.search(r"Game time:\s+([\d.]+)s", out)
    st = re.search(r"Steps:\s+(\d+)", out)
    if gt:
        game_time = float(gt.group(1))
    if st:
        steps = int(st.group(1))

    components: Dict[str, float] = {}
    in_comp_block = False
    for line in out.splitlines():
        if "Accumulated cost components" in line or re.match(r"^\s+\w+\s+:\s+[-\d.]+\s*$", line):
            in_comp_block = True
        if in_comp_block:
            cm = re.match(r"^\s+(\w+)\s+:\s+([-\d.]+)\s*$", line)
            if cm:
                components[cm.group(1)] = float(cm.group(2))

    return {
        "id": game_id,
        "result": result,
        "reason": reason,
        "game_time": game_time,
        "steps": steps,
        "wall_time": wall,
        "components": components,
        "returncode": proc.returncode,
    }


def aggregate(results: List[dict]) -> None:
    n = len(results)
    by_result = defaultdict(int)
    for r in results:
        by_result[r["result"]] += 1

    print()
    print("=" * 72)
    print(f"PARALLEL RUN SUMMARY ({n} games)")
    print("=" * 72)

    print("Outcomes:")
    for k in ("WIN", "LOSS", "TIMEOUT", "ERROR"):
        if by_result[k]:
            print(f"  {k:8s}: {by_result[k]} ({100*by_result[k]/n:.0f}%)")

    game_times = [r["game_time"] for r in results if r["game_time"] is not None]
    wall_times = [r["wall_time"] for r in results]
    steps = [r["steps"] for r in results if r["steps"] is not None]
    if game_times:
        print(f"\nGame time (s):  mean={np.mean(game_times):5.1f}  "
              f"min={min(game_times):4.1f}  max={max(game_times):4.1f}")
    if wall_times:
        print(f"Wall time (s):  mean={np.mean(wall_times):5.1f}  "
              f"min={min(wall_times):4.1f}  max={max(wall_times):4.1f}")
    if steps:
        print(f"Steps        :  mean={np.mean(steps):5.0f}  "
              f"min={min(steps):4d}  max={max(steps):4d}")

    print("\nPer-component cost (mean ± std across games):")
    all_comps = defaultdict(list)
    for r in results:
        for k, v in r["components"].items():
            all_comps[k].append(v)
    for k in sorted(all_comps):
        vs = all_comps[k]
        if len(vs) >= 2:
            print(f"  {k:25s}: {np.mean(vs):+10.2f} ± {np.std(vs):8.2f}    n={len(vs)}")
        else:
            print(f"  {k:25s}: {vs[0]:+10.2f}                           n=1")

    # Per-game one-line summary
    print("\nPer-game:")
    for r in sorted(results, key=lambda x: x["id"]):
        gt = f"{r['game_time']:5.1f}s" if r["game_time"] else "  ---"
        wt = f"{r['wall_time']:5.1f}s"
        print(f"  game {r['id']:2d}: {r['result']:8s} game={gt} wall={wt}  "
              f"reason: {r['reason']}")
    print("=" * 72)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    script = sys.argv[2] if len(sys.argv) > 2 else "scenario_circle.py"
    timeout = int(os.environ.get("RUN_TIMEOUT", "600"))
    sc2path = os.environ.get("SC2PATH", SC2PATH_DEFAULT)

    print(f"Launching {n_games} parallel games of {script}")
    print(f"  SC2PATH={sc2path}, timeout={timeout}s per game")
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=n_games) as ex:
        futures = {ex.submit(run_one, i, script, sc2path, timeout): i
                   for i in range(n_games)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
            except subprocess.TimeoutExpired:
                r = {"id": i, "result": "ERROR", "reason": "subprocess timeout",
                     "game_time": None, "steps": None,
                     "wall_time": timeout, "components": {}, "returncode": -1}
            except Exception as e:
                r = {"id": i, "result": "ERROR", "reason": f"exception: {e}",
                     "game_time": None, "steps": None,
                     "wall_time": time.time() - t0, "components": {}, "returncode": -1}
            results.append(r)
            print(f"  [game {i:2d}] {r['result']:8s} game_time={r['game_time']} "
                  f"wall={r['wall_time']:.1f}s")

    print(f"\nTotal wall time for {n_games} parallel games: {time.time() - t0:.1f}s")
    aggregate(results)


if __name__ == "__main__":
    main()
