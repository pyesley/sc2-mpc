"""
Cost function for Medivac drop micro.
Manages the hit-and-run cycle: approach → drop → shoot → pickup → retreat.

Key behaviors:
- When loaded: approach enemies, drop at range ~5 from priority target
- When fighting: maximize DPS, avoid zealot melee, track pickup timing
- When hurt or zealot close: pickup and retreat, boost away
- Overall: kill enemies with minimal marine losses
"""

import numpy as np
from typing import Tuple, Dict

MARINE_RANGE = 5.0


def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components = {}

    # ── 1. Enemy HP ──
    if state.stalker_alive:
        components['stalker_hp'] = 18.0 * (state.stalker_hp / 160.0)
    else:
        components['stalker_hp'] = 0.0

    if state.zealot_alive:
        components['zealot_hp'] = 12.0 * (state.zealot_hp / 150.0)
    else:
        components['zealot_hp'] = 0.0

    if not state.stalker_alive and not state.zealot_alive:
        components['stalker_hp'] = -10.0  # bonus for winning

    # ── 2. Marine survival ──
    alive = sum(1 for hp in state.marine_hps if hp > 0)
    total_hp = sum(max(0, hp) for hp in state.marine_hps)
    if alive == 0 and not state.marines_loaded:
        components['marine_survival'] = 50.0
    else:
        components['marine_survival'] = 8.0 * (1.0 - total_hp / (state.n_marines * 45.0))
        components['marine_survival'] += 10.0 * (state.n_marines - alive)

    # ── 3. Phase management ──
    enemies_alive = state.zealot_alive or state.stalker_alive

    if state.marines_loaded and enemies_alive:
        # Loaded: should be approaching to drop
        d = state.dist_medivac_enemies
        if d > 10.0:
            components['phase'] = 3.0 * (d - 10.0)  # too far, approach
        elif d > 5.0:
            # Good approach range — reward getting closer
            components['phase'] = 1.0 * (d - 5.0)
        elif d > 3.0:
            # Drop zone! Should drop here
            components['phase'] = -4.0  # reward being in drop zone
        else:
            # Too close while loaded — dangerous
            components['phase'] = 6.0 * (3.0 - d)

    elif not state.marines_loaded and enemies_alive:
        # Fighting: marines on ground
        # Check zealot proximity to marines
        z_dists = []
        for i in range(state.n_marines):
            if state.marine_hps[i] > 0:
                d = np.linalg.norm(state.marine_positions[i] - state.zealot_pos)
                z_dists.append(d)

        if z_dists and state.zealot_alive:
            min_z_dist = min(z_dists)
            if min_z_dist < 2.0:
                # Zealot in melee — need to pickup NOW
                components['phase'] = 12.0 * (2.0 - min_z_dist)
            elif min_z_dist < 4.0:
                # Zealot approaching — consider pickup soon
                components['phase'] = 2.0 * (4.0 - min_z_dist)
            else:
                # Safe, keep fighting
                components['phase'] = -3.0  # reward for being on ground shooting
        else:
            components['phase'] = -3.0
    else:
        components['phase'] = 0.0

    # ── 4. DPS uptime ──
    if not state.marines_loaded and enemies_alive:
        # Marines should be in range and shooting
        n_in_range = 0
        for i in range(state.n_marines):
            if state.marine_hps[i] <= 0:
                continue
            if state.stalker_alive:
                d = np.linalg.norm(state.marine_positions[i] - state.stalker_pos)
            elif state.zealot_alive:
                d = np.linalg.norm(state.marine_positions[i] - state.zealot_pos)
            else:
                d = 999
            if d <= MARINE_RANGE:
                n_in_range += 1
        components['dps'] = -3.0 * n_in_range + 2.0 * max(0, alive - n_in_range)
    else:
        components['dps'] = 2.0 if enemies_alive else 0.0

    # ── 5. Medivac safety ──
    # Medivac should not be in stalker range when marines are loaded
    if state.marines_loaded and state.stalker_alive:
        d_stalker = np.linalg.norm(state.medivac_pos - state.stalker_pos)
        if d_stalker < 6.0:  # stalker range
            components['medivac_safety'] = 5.0 * (6.0 - d_stalker)
        else:
            components['medivac_safety'] = 0.0
    else:
        components['medivac_safety'] = 0.0

    # ── 6. Boost management ──
    # Reward having boost available for escape
    if state.boost_available and state.marines_loaded:
        components['boost'] = -1.0  # good to have boost ready
    elif state.boost_active and state.marines_loaded:
        components['boost'] = -2.0  # boosting with marines = approaching fast
    else:
        components['boost'] = 0.0

    total = sum(components.values())
    return total, components
