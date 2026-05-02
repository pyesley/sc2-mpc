"""
Role-free cost function for 2 Marines vs 1 Zealot MPC.
Eureka-style: returns (total_cost, components_dict).

VERSION 4: Role-free design
Instead of hardcoded bait/shooter, the cost function dynamically labels
marines as 'near' (closer to zealot) and 'far' (farther from zealot).

Near marine: should maintain d=3.0-4.5, kite tangentially, hold aggro.
Far marine: should maintain d=4.5-5.5, stay stationary, shoot.
Key constraint: near and far distances must be separated by >= 2.0 units
so the zealot has a clear target and doesn't switch randomly.
"""

import numpy as np
from typing import Tuple, Dict


def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components = {}

    d1 = state.dist_m1_zealot
    d2 = state.dist_m2_zealot

    # Dynamically assign near/far
    if d1 <= d2:
        near_dist = d1
        far_dist = d2
        near_hp = state.m1_hp
        far_hp = state.m2_hp
        near_pos = np.array(state.m1_pos)
        far_pos = np.array(state.m2_pos)
        near_weapon_ready = state.m1_weapon_ready
        far_weapon_ready = state.m2_weapon_ready
    else:
        near_dist = d2
        far_dist = d1
        near_hp = state.m2_hp
        far_hp = state.m1_hp
        near_pos = np.array(state.m2_pos)
        far_pos = np.array(state.m1_pos)
        near_weapon_ready = state.m2_weapon_ready
        far_weapon_ready = state.m1_weapon_ready

    zealot_pos = np.array(state.zealot_pos)

    # ── 1. Zealot HP: want it dead ──
    zealot_hp_frac = state.zealot_hp / state.zealot_hp_max
    components['zealot_hp'] = 12.0 * zealot_hp_frac

    # ── 2. Marine survival ──
    if near_hp > 0:
        components['near_survival'] = 5.0 * np.exp(-3.0 * (near_hp / 45.0))
    else:
        components['near_survival'] = 25.0

    if far_hp > 0:
        components['far_survival'] = 6.0 * (1.0 - far_hp / 45.0)
    else:
        components['far_survival'] = 40.0

    # ── 3. Distance separation (aggro clarity) ──
    # The near marine must be clearly closer so the zealot has one target.
    # Required: far_dist - near_dist >= 2.0
    separation = far_dist - near_dist
    if separation > 3.0:
        components['dist_separation'] = -4.0  # strong reward
    elif separation > 2.0:
        components['dist_separation'] = -2.0
    elif separation > 1.0:
        components['dist_separation'] = 5.0 * (2.0 - separation)
    elif separation > 0:
        components['dist_separation'] = 12.0 * (1.0 - separation)
    else:
        # Both at same distance — ambiguous aggro
        components['dist_separation'] = 20.0

    # ── 4. Near marine distance (kiter) ──
    # Must stay 3.0-4.5: close enough for aggro, far enough to not die.
    # Marine & zealot have same speed, so below 2.5 = guaranteed damage.
    if near_dist < 1.5:
        # Capped exponential — still very bad, but doesn't dwarf all other costs
        components['near_dist'] = min(25.0 * (1.5 - near_dist) + 10.0, 50.0)
    elif near_dist < 2.5:
        components['near_dist'] = 10.0 * (2.5 - near_dist)
    elif near_dist < 3.0:
        components['near_dist'] = 4.0 * (3.0 - near_dist)
    elif near_dist <= 4.5:
        # Sweet spot
        components['near_dist'] = 0.2 * abs(near_dist - 3.5)
    elif near_dist <= 6.0:
        components['near_dist'] = 2.0 * (near_dist - 4.5)
    else:
        components['near_dist'] = 5.0 * (near_dist - 6.0)

    # ── 5. Far marine distance (shooter) ──
    # Should be at 4.5-5.5 (in range 5, well away from melee)
    if far_dist < 2.0:
        components['far_dist'] = 25.0 * (2.0 - far_dist)
    elif far_dist < 3.5:
        components['far_dist'] = 8.0 * (3.5 - far_dist)
    elif far_dist <= 5.0:
        # Good shooting range
        components['far_dist'] = 0.2 * abs(far_dist - 4.8)
    elif far_dist <= 6.0:
        components['far_dist'] = 2.0 * (far_dist - 5.0)
    else:
        components['far_dist'] = 4.0 * (far_dist - 6.0)

    # ── 6. Far marine DPS uptime ──
    # Far marine should be stationary and in range to maximize damage
    if far_weapon_ready and far_dist <= 5.0:
        components['far_dps'] = -6.0
    elif far_dist <= 5.0:
        components['far_dps'] = -2.0
    else:
        components['far_dps'] = 5.0

    # ── 7. Marine physical separation ──
    # Don't bunch up (zealot AoE splash from targeting), don't spread too far
    marine_sep = np.linalg.norm(near_pos - far_pos)
    if marine_sep < 3.0:
        components['bunching'] = 4.0 * (3.0 - marine_sep)
    elif marine_sep > 12.0:
        components['bunching'] = 1.5 * (marine_sep - 12.0)
    else:
        components['bunching'] = 0.0

    # ── 8. Near marine angular offset ──
    # Near marine should not be between zealot and far marine.
    # Being off to the side means zealot paths away from the shooter.
    if near_dist > 0.5 and far_dist > 0.5:
        zn = near_pos - zealot_pos
        zf = far_pos - zealot_pos
        zn_norm = np.linalg.norm(zn)
        zf_norm = np.linalg.norm(zf)
        if zn_norm > 0.1 and zf_norm > 0.1:
            cos_angle = np.clip(np.dot(zn, zf) / (zn_norm * zf_norm), -1, 1)
            angle = np.arccos(cos_angle)
            if angle > np.pi / 3:
                components['angle'] = -2.0  # reward: near is off to the side
            elif angle > np.pi / 6:
                components['angle'] = 0.0
            else:
                components['angle'] = 3.0 * (np.pi / 6 - angle)
        else:
            components['angle'] = 0.0
    else:
        components['angle'] = 0.0

    total_cost = sum(components.values())
    return total_cost, components
