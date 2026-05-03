"""
Bio cost — variant v1b: MARAUDER-FRONT FORMATION.

Hypothesis: marauders are tankier (125 HP, 1 armor) than marines
(45 HP). If marauders are between the marines and the stalkers, the
stalkers prefer-target marauders (closest), and marines deal damage
unmolested. Currently the cost has no notion of formation — marines
end up exposed because no one tells them to stay BEHIND marauders.

New component: 'marauder_front'. For each alive marine, compare its
distance to nearest alive stalker against the nearest alive marauder's
distance to that same stalker. If the marine is CLOSER, it's in front
of the marauder shield → penalty. Sum over all (marine, stalker) pairs.

Plugs into cost_bio via EXTRA_COST_FN_BATCH hook so the rest of the
composition is unchanged.
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio

W_MARAUDER_FRONT = 4.0


def _marauder_front_batch(ctx):
    """Penalize alive marines that are closer to a stalker than any alive
    marauder is to that stalker."""
    m_pos = ctx['m_pos']           # (N, n_m, 2)
    mm_pos = ctx['mm_pos']         # (N, n_mm, 2)
    s_pos = ctx['s_pos']           # (N, n_s, 2)
    m_alive = ctx['m_alive']       # (N, n_m)
    mm_alive = ctx['mm_alive']     # (N, n_mm)
    s_alive = ctx['s_alive']       # (N, n_s)
    active = ctx['active']         # (N,)

    # Distances marine→stalker, marauder→stalker
    m2s = np.linalg.norm(m_pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1)
    mm2s = np.linalg.norm(mm_pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1)

    # Mask dead stalkers (∞) and dead marauders (∞)
    m2s = np.where(s_alive[:, None, :], m2s, np.inf)
    mm2s_eff = np.where(mm_alive[:, :, None] & s_alive[:, None, :], mm2s, np.inf)

    # Per stalker: closest alive marauder distance (N, n_s)
    mm_min = np.min(mm2s_eff, axis=1)

    # Per (marine, stalker): excess = marine_d - marauder_d
    # If positive (marine further than marauder), good. Negative = bad (marine in front).
    excess = m2s - mm_min[:, None, :]

    # Penalty for marines IN FRONT (excess < 0), only if marine alive and stalker alive
    in_front = (excess < 0.0) & m_alive[:, :, None] & s_alive[:, None, :]
    penalty = np.where(in_front, -excess, 0.0)   # how far in front
    total = W_MARAUDER_FRONT * penalty.sum(axis=(1, 2))    # (N,)
    total = np.where(active, total, 0.0)
    return {'marauder_front': total}


def _marauder_front_single(state):
    """Single-state mirror of the batched version (for spec / testing)."""
    s_alive = list(state.stalker_alive)
    m_hps = list(state.marine_hps)
    mm_hps = list(state.marauder_hps)
    if not any(s_alive):
        return {'marauder_front': 0.0}
    if not any(h > 0 for h in m_hps) or not any(h > 0 for h in mm_hps):
        return {'marauder_front': 0.0}

    total = 0.0
    for sj in range(state.n_stalkers):
        if not s_alive[sj]:
            continue
        sp = np.asarray(state.stalker_positions[sj])
        # Closest alive marauder
        mm_d = [np.linalg.norm(np.asarray(state.marauder_positions[i]) - sp)
                for i in range(state.n_marauders) if mm_hps[i] > 0]
        if not mm_d:
            continue
        mm_min = min(mm_d)
        # Each alive marine: penalty if closer to this stalker than the marauder
        for mi in range(state.n_marines):
            if m_hps[mi] <= 0:
                continue
            mp = np.asarray(state.marine_positions[mi])
            d = float(np.linalg.norm(mp - sp))
            if d < mm_min:
                total += (mm_min - d)
    return {'marauder_front': W_MARAUDER_FRONT * total}


cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
