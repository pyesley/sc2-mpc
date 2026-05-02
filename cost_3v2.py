"""
Cost function for 3 Marines vs Zealot + Stalker.
Two-phase fight:
  Phase 1 (stalker alive): focus-fire stalker, stay out of zealot melee
  Phase 2 (stalker dead): kite zealot with remaining marines

Key insight: stalker outranges (6 vs 5) and outruns (2.95 vs 2.25) marines.
Marines must ENGAGE the stalker, not run from it. All 3 must focus-fire.
"""

import numpy as np
from typing import Tuple, Dict

MARINE_RANGE = 5.0
STALKER_RANGE = 6.0


def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components = {}
    n = state.n_marines
    live = [i for i in range(n) if state.marine_hps[i] > 0]
    n_alive = len(live)

    # ── 1. Enemy HP: priority target matters ──
    if state.stalker_alive:
        # Phase 1: stalker is priority. Reward stalker damage heavily.
        stalker_frac = state.stalker_hp / 160.0  # 80+80
        components['stalker_hp'] = 20.0 * stalker_frac
        # Zealot HP matters less in phase 1
        if state.zealot_alive:
            zealot_frac = state.zealot_hp / 150.0
            components['zealot_hp'] = 3.0 * zealot_frac
        else:
            components['zealot_hp'] = 0.0
    else:
        components['stalker_hp'] = 0.0
        if state.zealot_alive:
            zealot_frac = state.zealot_hp / 150.0
            components['zealot_hp'] = 15.0 * zealot_frac
        else:
            components['zealot_hp'] = -10.0  # both dead = reward

    # ── 2. Marine survival ──
    total_hp = sum(state.marine_hps[i] for i in live)
    max_hp = n * 45.0
    if n_alive == 0:
        components['marine_survival'] = 60.0
    else:
        components['marine_survival'] = 10.0 * (1.0 - total_hp / max_hp)
        # Extra penalty per dead marine
        n_dead = n - n_alive
        components['marine_survival'] += 12.0 * n_dead

    # ── 3. Positioning relative to stalker (Phase 1) ──
    if state.stalker_alive and live:
        dists_to_stalker = [np.linalg.norm(state.marine_positions[i] - state.stalker_pos)
                            for i in live]
        avg_dist_stalker = np.mean(dists_to_stalker)

        # Marines should be in their attack range of stalker (d <= 5.0)
        # But stalker outranges them (6), so marines MUST close to 5.0
        n_in_range = sum(1 for d in dists_to_stalker if d <= MARINE_RANGE)
        n_out_range = n_alive - n_in_range

        # Reward marines in range, penalize out of range
        components['stalker_engage'] = -3.0 * n_in_range + 6.0 * n_out_range

        # Penalty for being too far (can't shoot stalker)
        if avg_dist_stalker > MARINE_RANGE + 1:
            components['stalker_approach'] = 4.0 * (avg_dist_stalker - MARINE_RANGE)
        else:
            components['stalker_approach'] = 0.0
    else:
        components['stalker_engage'] = 0.0
        components['stalker_approach'] = 0.0

    # ── 4. Zealot avoidance ──
    if state.zealot_alive and live:
        dists_to_zealot = [np.linalg.norm(state.marine_positions[i] - state.zealot_pos)
                           for i in live]

        # Penalty for marines close to zealot (melee = death)
        danger_cost = 0.0
        for d in dists_to_zealot:
            if d < 1.5:
                danger_cost += 15.0 * (1.5 - d)
            elif d < 3.0:
                danger_cost += 3.0 * (3.0 - d)
        components['zealot_danger'] = danger_cost

        # In phase 2 (stalker dead), also manage kiting distance
        if not state.stalker_alive:
            min_zealot_dist = min(dists_to_zealot)
            if min_zealot_dist < 2.0:
                components['zealot_kite'] = 10.0 * (2.0 - min_zealot_dist)
            elif min_zealot_dist > 6.0:
                components['zealot_kite'] = 2.0 * (min_zealot_dist - 6.0)
            else:
                components['zealot_kite'] = 0.2 * abs(min_zealot_dist - 3.5)
        else:
            components['zealot_kite'] = 0.0
    else:
        components['zealot_danger'] = 0.0
        components['zealot_kite'] = 0.0

    # ── 5. Focus fire: marines should all shoot the same target ──
    if live and state.stalker_alive:
        # Reward marines being at similar distance to stalker (focus fire)
        dists = [np.linalg.norm(state.marine_positions[i] - state.stalker_pos) for i in live]
        if len(dists) > 1:
            spread = np.std(dists)
            components['focus_fire'] = 2.0 * spread
        else:
            components['focus_fire'] = 0.0
    else:
        components['focus_fire'] = 0.0

    # ── 6. Marine spread (don't stack up for AoE/splash) ──
    if len(live) >= 2:
        min_sep = float('inf')
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                sep = np.linalg.norm(
                    state.marine_positions[live[i]] - state.marine_positions[live[j]])
                min_sep = min(min_sep, sep)
        if min_sep < 2.0:
            components['marine_spread'] = 3.0 * (2.0 - min_sep)
        else:
            components['marine_spread'] = 0.0
    else:
        components['marine_spread'] = 0.0

    # ── 7. DPS uptime ──
    dps_reward = 0.0
    for i in live:
        if state.marine_weapon_ready[i]:
            # Check if in range of priority target
            if state.stalker_alive:
                d = np.linalg.norm(state.marine_positions[i] - state.stalker_pos)
            elif state.zealot_alive:
                d = np.linalg.norm(state.marine_positions[i] - state.zealot_pos)
            else:
                d = 999
            if d <= MARINE_RANGE:
                dps_reward -= 3.0  # reward for being ready to fire in range
    components['dps_uptime'] = dps_reward

    total = sum(components.values())
    return total, components
