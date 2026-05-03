"""
Bio cost — variant v3b: STATE-CONDITIONAL MODE MIX.

Hypothesis: there's no single best mode mix — it depends on the
phase of the fight.
  - At full HP (opening): HOLD + SHIELD heavy, low FOCUS_FIRE.
    Setup formation before contact.
  - At moderate damage (mid-fight): SHIELD + FOCUS_FIRE heavy, low HOLD.
    Marauders shielding while bio focuses lowest-HP enemy.
  - At heavy damage (late, finishing or losing): FOCUS_FIRE heavy
    + RETREAT some. Either burst the kill or back off.

Plus v1b's marauder_front hook to keep formation rewarded.
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch

# Mode order: INDEPENDENT, RETREAT, FOCUS_FIRE, SHIELD, HOLD
def _state_conditional_mode_probs(state):
    # Compute current bio HP fraction
    m_hp = sum(h for h in state.marine_hps if h > 0)
    mm_hp = sum(h for h in state.marauder_hps if h > 0)
    bio_hp = m_hp + mm_hp
    bio_max = state.n_marines * 45.0 + state.n_marauders * 125.0
    frac = bio_hp / max(bio_max, 1.0)
    if frac > 0.95:
        # Opening — set up formation, hold
        return np.array([0.10, 0.08, 0.10, 0.32, 0.40])
    elif frac > 0.65:
        # Mid-fight — shield + focus damage
        return np.array([0.14, 0.10, 0.30, 0.32, 0.14])
    else:
        # Heavy damage — burst or back off
        return np.array([0.10, 0.18, 0.42, 0.20, 0.10])

mpc_vectorized_bio.MODE_PROBS = _state_conditional_mode_probs
