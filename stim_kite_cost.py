"""
Cost function for 1 Stimmed Marine vs 1 Zealot stutter-step kiting.
Eureka-style: returns (total_cost, components_dict).

VERSION 1: Initial design
Key behaviors to encode:
  - Stutter-step: STOP to shoot when weapon ready + in range, MOVE during cooldown
  - Maintain range 4-5 from zealot (close enough to shoot, far enough for safety)
  - Stim management: stim early, don't double-stim, don't stim at low HP
  - Maximize DPS uptime (minimize time spent out of range or moving when could shoot)
"""

import numpy as np
from typing import Tuple, Dict

MARINE_RANGE = 5.0


def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components = {}
    dist = state.dist

    # ── 1. Zealot HP: want it dead ──
    zealot_hp_frac = state.zealot_hp / state.zealot_hp_max
    components['zealot_hp'] = 15.0 * zealot_hp_frac

    # ── 2. Marine survival ──
    if state.marine_hp > 0:
        hp_frac = state.marine_hp / 45.0
        components['marine_survival'] = 8.0 * np.exp(-3.0 * hp_frac)
    else:
        components['marine_survival'] = 50.0

    # ── 3. Distance management ──
    # With stim (speed 4.73 vs 2.25), marine can easily maintain range.
    # Optimal: 4.0-5.0 (shoot range, safe from melee)
    # Without stim: same speed as zealot, need to be further
    if state.is_stimmed:
        optimal_dist = 4.5
        if dist < 1.5:
            components['distance'] = 30.0 * (1.5 - dist)
        elif dist < 3.0:
            components['distance'] = 8.0 * (3.0 - dist)
        elif dist <= 5.0:
            components['distance'] = 0.3 * abs(dist - optimal_dist)
        elif dist <= 6.0:
            components['distance'] = 2.0 * (dist - 5.0)
        else:
            components['distance'] = 5.0 * (dist - 6.0)
    else:
        # Without stim, same speed — need more distance buffer
        optimal_dist = 4.5
        if dist < 2.0:
            components['distance'] = 40.0 * (2.0 - dist)
        elif dist < 3.5:
            components['distance'] = 10.0 * (3.5 - dist)
        elif dist <= 5.0:
            components['distance'] = 0.3 * abs(dist - optimal_dist)
        elif dist <= 6.0:
            components['distance'] = 1.5 * (dist - 5.0)
        else:
            components['distance'] = 4.0 * (dist - 6.0)

    # ── 4. Stutter-step timing ──
    # CRITICAL: stop to shoot when weapon ready + in range, move otherwise
    if state.weapon_ready and dist <= MARINE_RANGE:
        # Should be stationary (shooting) — reward for being weapon_ready in range
        components['stutter_step'] = -6.0  # reward
    elif not state.weapon_ready and dist < 4.0:
        # Weapon on cooldown, too close — should be moving away
        components['stutter_step'] = 3.0 * (4.0 - dist)
    elif not state.weapon_ready and dist <= MARINE_RANGE:
        # Weapon on cooldown, at range — OK, can wait or adjust
        components['stutter_step'] = 0.0
    else:
        # Out of range — penalty
        components['stutter_step'] = 3.0

    # ── 5. Stim management ──
    if state.is_stimmed:
        # Good — being stimmed is rewarded
        components['stim'] = -3.0
        # But warn if stim is about to expire and zealot is close
        if state.stim_remaining < 2.0 and dist < 4.0:
            components['stim'] += 4.0  # danger: stim expiring while close
    else:
        # Not stimmed
        if state.marine_hp > 20:
            # Should stim — penalty for not being stimmed when healthy
            components['stim'] = 5.0
        else:
            # Low HP — stimming would be risky
            components['stim'] = 0.0

    # ── 6. DPS efficiency ──
    # Reward being in a position to deal damage
    if dist <= MARINE_RANGE and not state.weapon_ready:
        # In range, weapon cycling — good, damage being dealt
        components['dps'] = -2.0
    elif dist <= MARINE_RANGE and state.weapon_ready:
        # In range and ready to fire — should shoot (stop moving)
        components['dps'] = -4.0
    else:
        # Out of range
        components['dps'] = 3.0

    total_cost = sum(components.values())
    return total_cost, components
