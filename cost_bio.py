"""
Compositional cost for Bio + Medivac vs Mixed Protoss.

Composes type-pair primitives from cost_primitives.py over an army of
mixed types. Adds bio-specific globals: stalker priority, medivac
heal-vs-safety tradeoff, focus-fire on lowest-HP enemy.

State (BioState in scenario_bio.py) holds positions / HPs / alive
flags as arrays per type. n_marines, n_marauders, n_medivacs (=1),
n_zealots, n_stalkers are fixed at scenario setup.

Both single-state (compute_cost — spec / Eureka iteration) and batched
(compute_cost_batch — used by mpc_vectorized_bio.py) forms are kept
in sync in this file.
"""

import numpy as np
from typing import Tuple, Dict

from cost_primitives import (
    MARINE_RANGE, MARAUDER_RANGE, STALKER_RANGE, MEDIVAC_HEAL_RANGE,
    MARINE_HP_MAX, MARAUDER_HP_MAX, MEDIVAC_HP_MAX,
    ZEALOT_HP_MAX, STALKER_HP_MAX,
    pairwise_dist,
    kite_marine_vs_zealot, engage_marine_vs_stalker,
    kite_marauder_vs_zealot, engage_marauder_vs_stalker,
    medivac_safety, medivac_heal_proximity,
)

# ─────────────────────────────────────────────────────────────
# Weight knobs — variants tune these via module attribute override.
# Functions read these at call time, so subprocess-level overrides
# (e.g. cost_bio_v1a.py setting cost_bio.W_DEAD_MARINE = 24.0) take
# effect for the whole process without code duplication.
# ─────────────────────────────────────────────────────────────
W_ENEMY_HP = 14.0
W_WIN_BONUS = -25.0

W_SURV_M = 6.0           # marine HP loss
W_SURV_MM = 8.0          # marauder HP loss
W_SURV_MV = 10.0         # medivac HP loss
W_DEAD_M = 8.0           # per-marine death
W_DEAD_MM = 14.0         # per-marauder death
W_DEAD_MV = 40.0         # per-medivac death
W_WIPEOUT = 100.0        # all bio + medivac dead

W_MATCHUP_M = 1.0        # multiplier on per-marine matchup primitive sum
W_MATCHUP_MM = 1.0       # multiplier on per-marauder matchup primitive sum
W_MEDIVAC_SAFETY = 1.0   # multiplier
W_MEDIVAC_HEAL = 1.0     # multiplier

W_FOCUS_FIRE = 2.5       # per attacker in range of weakest enemy (negated)
W_STALKER_PRIORITY = 3.0 # per (stalker, distance over MARAUDER_RANGE)
W_DPS_M = 2.0            # reward per stationary marine in range
W_DPS_MM = 2.5           # reward per stationary marauder in range
W_COHESION = 0.0         # per (max-pairwise-bio-distance - COHESION_THRESHOLD).
                          # Default 0 (off). Variants set this to penalize
                          # bio army getting split (the "divide and conquer"
                          # failure mode where enemy peels off a unit).
COHESION_THRESHOLD = 6.0 # bio max pairwise dist allowed before cohesion penalty

# Optional structural hooks — variants can plug in extra cost components
# without rewriting the whole composer. Each callable returns dict[str,
# float] (single-state) or dict[str, ndarray(N,)] (batched), summed
# straight into the existing `components` map.
EXTRA_COST_FN = None         # f(state) -> dict[str, float]
EXTRA_COST_FN_BATCH = None   # f(traj_t, ctx) -> dict[str, ndarray(N,)]
                              # called inside the per-timestep loop with
                              # the per-timestep dict and a context dict
                              # of useful precomputed arrays


# ─────────────────────────────────────────────────────────────
# Single-state form (spec / readable)
# ─────────────────────────────────────────────────────────────
def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components: Dict[str, float] = {}

    # Alive masks
    m_alive = np.asarray(state.marine_hps) > 0
    mm_alive = np.asarray(state.marauder_hps) > 0
    mv_alive = np.asarray(state.medivac_hps) > 0
    z_alive = np.asarray(state.zealot_alive)
    s_alive = np.asarray(state.stalker_alive)

    n_m = int(m_alive.sum())
    n_mm = int(mm_alive.sum())
    n_mv = int(mv_alive.sum())
    n_z = int(z_alive.sum())
    n_s = int(s_alive.sum())

    # ── 1. Enemy HP drive (combined zealot + stalker pool) ──
    z_hp = np.asarray(state.zealot_hps) * z_alive
    s_hp = np.asarray(state.stalker_hps) * s_alive
    total_e_hp = float(z_hp.sum() + s_hp.sum())
    total_e_hp_max = state.n_zealots * ZEALOT_HP_MAX + state.n_stalkers * STALKER_HP_MAX
    if n_z + n_s == 0:
        components['enemy_hp'] = W_WIN_BONUS
    else:
        components['enemy_hp'] = W_ENEMY_HP * (total_e_hp / total_e_hp_max)

    # ── 2. Survival (per-type weighted by unit value) ──
    m_hp_sum = float(np.sum(np.asarray(state.marine_hps) * m_alive))
    mm_hp_sum = float(np.sum(np.asarray(state.marauder_hps) * mm_alive))
    mv_hp_sum = float(np.sum(np.asarray(state.medivac_hps) * mv_alive))

    survival = 0.0
    survival += W_SURV_M * (1.0 - m_hp_sum / max(state.n_marines * MARINE_HP_MAX, 1e-6))
    survival += W_SURV_MM * (1.0 - mm_hp_sum / max(state.n_marauders * MARAUDER_HP_MAX, 1e-6))
    survival += W_SURV_MV * (1.0 - mv_hp_sum / max(state.n_medivacs * MEDIVAC_HP_MAX, 1e-6))
    survival += W_DEAD_M * (state.n_marines - n_m)
    survival += W_DEAD_MM * (state.n_marauders - n_mm)
    survival += W_DEAD_MV * (state.n_medivacs - n_mv)
    if n_m + n_mm == 0 and n_mv == 0:
        survival += W_WIPEOUT
    components['survival'] = survival

    # If combat is over, skip the spatial components
    if (n_m + n_mm + n_mv == 0) or (n_z + n_s == 0):
        for k in ('matchup_marines', 'matchup_marauders', 'medivac_safety',
                  'medivac_heal', 'focus_fire', 'stalker_priority',
                  'dps_uptime'):
            components[k] = 0.0
        return sum(components.values()), components

    m_pos = np.asarray(state.marine_positions)
    mm_pos = np.asarray(state.marauder_positions)
    mv_pos = np.asarray(state.medivac_positions)
    z_pos = np.asarray(state.zealot_positions)
    s_pos = np.asarray(state.stalker_positions)

    # ── 3. Marine matchup (vs nearest alive zealot / stalker) ──
    matchup_m = 0.0
    if n_m > 0:
        for i in range(state.n_marines):
            if not m_alive[i]:
                continue
            d_zs = np.array([np.linalg.norm(m_pos[i] - z_pos[j])
                              for j in range(state.n_zealots) if z_alive[j]])
            d_ss = np.array([np.linalg.norm(m_pos[i] - s_pos[j])
                              for j in range(state.n_stalkers) if s_alive[j]])
            d_z = float(d_zs.min()) if len(d_zs) else 1e6
            d_s = float(d_ss.min()) if len(d_ss) else 1e6
            # If a zealot is in immediate melee threat, kite it; otherwise
            # engage the priority target (stalker if alive, else zealot)
            if d_z < 2.5:
                matchup_m += float(kite_marine_vs_zealot(d_z))
            elif n_s > 0:
                matchup_m += float(engage_marine_vs_stalker(d_s))
            else:
                matchup_m += float(kite_marine_vs_zealot(d_z))
    components['matchup_marines'] = W_MATCHUP_M * matchup_m

    # ── 4. Marauder matchup ──
    matchup_mm = 0.0
    if n_mm > 0:
        for i in range(state.n_marauders):
            if not mm_alive[i]:
                continue
            d_zs = np.array([np.linalg.norm(mm_pos[i] - z_pos[j])
                              for j in range(state.n_zealots) if z_alive[j]])
            d_ss = np.array([np.linalg.norm(mm_pos[i] - s_pos[j])
                              for j in range(state.n_stalkers) if s_alive[j]])
            d_z = float(d_zs.min()) if len(d_zs) else 1e6
            d_s = float(d_ss.min()) if len(d_ss) else 1e6
            if d_z < 2.5:
                matchup_mm += float(kite_marauder_vs_zealot(d_z))
            elif n_s > 0:
                matchup_mm += float(engage_marauder_vs_stalker(d_s))
            else:
                matchup_mm += float(kite_marauder_vs_zealot(d_z))
    components['matchup_marauders'] = W_MATCHUP_MM * matchup_mm

    # ── 5. Medivac safety ──
    if n_mv > 0:
        # Distance from medivac to nearest alive enemy (zealot+stalker)
        all_enemy_pos = []
        if n_z > 0:
            all_enemy_pos.extend(z_pos[j] for j in range(state.n_zealots) if z_alive[j])
        if n_s > 0:
            all_enemy_pos.extend(s_pos[j] for j in range(state.n_stalkers) if s_alive[j])
        d_min = float(min(np.linalg.norm(mv_pos[0] - p) for p in all_enemy_pos))
        components['medivac_safety'] = W_MEDIVAC_SAFETY * float(medivac_safety(d_min))
    else:
        components['medivac_safety'] = 0.0

    # ── 6. Medivac heal proximity (to most-injured bio ally) ──
    if n_mv > 0 and (n_m + n_mm > 0):
        bio_pos = []
        bio_def = []
        for i in range(state.n_marines):
            if m_alive[i]:
                bio_pos.append(m_pos[i])
                bio_def.append(MARINE_HP_MAX - state.marine_hps[i])  # damage taken
        for i in range(state.n_marauders):
            if mm_alive[i]:
                bio_pos.append(mm_pos[i])
                bio_def.append(MARAUDER_HP_MAX - state.marauder_hps[i])
        # Pick the most injured (largest damage)
        injured_idx = int(np.argmax(bio_def))
        d_inj = float(np.linalg.norm(mv_pos[0] - bio_pos[injured_idx]))
        # Only reward heal proximity if there's actually injury
        if max(bio_def) > 1.0:
            components['medivac_heal'] = W_MEDIVAC_HEAL * float(medivac_heal_proximity(d_inj))
        else:
            components['medivac_heal'] = 0.0
    else:
        components['medivac_heal'] = 0.0

    # ── 7. Focus fire bonus on lowest-HP alive enemy ──
    enemy_hps = []
    enemy_pos_list = []
    for j in range(state.n_zealots):
        if z_alive[j]:
            enemy_hps.append(state.zealot_hps[j])
            enemy_pos_list.append(z_pos[j])
    for j in range(state.n_stalkers):
        if s_alive[j]:
            enemy_hps.append(state.stalker_hps[j])
            enemy_pos_list.append(s_pos[j])
    if enemy_hps:
        weakest = int(np.argmin(enemy_hps))
        weakest_pos = enemy_pos_list[weakest]
        in_range = 0
        for i in range(state.n_marines):
            if m_alive[i] and np.linalg.norm(m_pos[i] - weakest_pos) <= MARINE_RANGE:
                in_range += 1
        for i in range(state.n_marauders):
            if mm_alive[i] and np.linalg.norm(mm_pos[i] - weakest_pos) <= MARAUDER_RANGE:
                in_range += 1
        components['focus_fire'] = -W_FOCUS_FIRE * in_range
    else:
        components['focus_fire'] = 0.0

    # ── 8. Stalker priority — penalize if no bio in stalker range ──
    if n_s > 0:
        sp_cost = 0.0
        for j in range(state.n_stalkers):
            if not s_alive[j]:
                continue
            min_d = 1e6
            for i in range(state.n_marines):
                if m_alive[i]:
                    min_d = min(min_d, np.linalg.norm(m_pos[i] - s_pos[j]))
            for i in range(state.n_marauders):
                if mm_alive[i]:
                    min_d = min(min_d, np.linalg.norm(mm_pos[i] - s_pos[j]))
            if min_d > MARAUDER_RANGE:
                sp_cost += W_STALKER_PRIORITY * (min_d - MARAUDER_RANGE)
        components['stalker_priority'] = sp_cost
    else:
        components['stalker_priority'] = 0.0

    # ── 9. DPS uptime — per stationary bio in range of any enemy ──
    dps = 0.0
    for i in range(state.n_marines):
        if m_alive[i] and state.marine_weapon_ready[i]:
            for p in enemy_pos_list:
                if np.linalg.norm(m_pos[i] - p) <= MARINE_RANGE:
                    dps -= W_DPS_M
                    break
    for i in range(state.n_marauders):
        if mm_alive[i] and state.marauder_weapon_ready[i]:
            for p in enemy_pos_list:
                if np.linalg.norm(mm_pos[i] - p) <= MARAUDER_RANGE:
                    dps -= W_DPS_MM
                    break
    components['dps_uptime'] = dps

    # ── 10. Army cohesion (anti-split). Penalize stragglers. ──
    coh = 0.0
    if W_COHESION > 0.0:
        bio_pos_alive = [p for i, p in enumerate(state.marine_positions) if m_alive[i]]
        bio_pos_alive += [p for i, p in enumerate(state.marauder_positions) if mm_alive[i]]
        if len(bio_pos_alive) >= 2:
            arr = np.asarray(bio_pos_alive)
            d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            max_pair = float(d.max())
            if max_pair > COHESION_THRESHOLD:
                coh = W_COHESION * (max_pair - COHESION_THRESHOLD)
    components['cohesion'] = coh

    # Optional structural extension (variants set EXTRA_COST_FN)
    if EXTRA_COST_FN is not None:
        for k, v in EXTRA_COST_FN(state).items():
            components[k] = components.get(k, 0.0) + float(v)

    return sum(components.values()), components


# ─────────────────────────────────────────────────────────────
# Batched form (used by mpc_vectorized_bio.py)
# ─────────────────────────────────────────────────────────────
def compute_cost_batch(traj, n_marines: int, n_marauders: int,
                       n_medivacs: int, n_zealots: int, n_stalkers: int):
    """Vectorized cost over (N, H) trajectories.

    traj keys (shapes):
      m_pos    (N, H, n_m, 2)        marine positions
      mm_pos   (N, H, n_mm, 2)       marauder positions
      mv_pos   (N, H, n_mv, 2)
      z_pos    (N, H, n_z, 2)
      s_pos    (N, H, n_s, 2)
      m_hp     (N, H, n_m)
      mm_hp    (N, H, n_mm)
      mv_hp    (N, H, n_mv)
      z_hp     (N, H, n_z)
      s_hp     (N, H, n_s)
      z_alive  (N, H, n_z)  bool
      s_alive  (N, H, n_s)  bool
      m_ready  (N, H, n_m)  bool
      mm_ready (N, H, n_mm) bool
    """
    N, H = traj['m_hp'].shape[:2]

    keys = ('enemy_hp', 'survival',
            'matchup_marines', 'matchup_marauders',
            'medivac_safety', 'medivac_heal',
            'focus_fire', 'stalker_priority', 'dps_uptime')
    comps = {k: np.zeros(N) for k in keys}

    err_ctx = np.errstate(invalid='ignore', over='ignore')
    err_ctx.__enter__()

    for t in range(H):
        m_pos = traj['m_pos'][:, t]    # (N, n_m, 2)
        mm_pos = traj['mm_pos'][:, t]
        mv_pos = traj['mv_pos'][:, t]
        z_pos = traj['z_pos'][:, t]
        s_pos = traj['s_pos'][:, t]
        m_hp = traj['m_hp'][:, t]
        mm_hp = traj['mm_hp'][:, t]
        mv_hp = traj['mv_hp'][:, t]
        z_hp = traj['z_hp'][:, t]
        s_hp = traj['s_hp'][:, t]
        z_alive = traj['z_alive'][:, t]
        s_alive = traj['s_alive'][:, t]
        m_ready = traj['m_ready'][:, t]
        mm_ready = traj['mm_ready'][:, t]

        m_alive = m_hp > 0
        mm_alive = mm_hp > 0
        mv_alive = mv_hp > 0
        n_z_alive = z_alive.sum(axis=-1)
        n_s_alive = s_alive.sum(axis=-1)
        n_e_alive = n_z_alive + n_s_alive
        n_bio_alive = m_alive.sum(axis=-1) + mm_alive.sum(axis=-1)
        n_mv_alive = mv_alive.sum(axis=-1)
        active = (n_bio_alive + n_mv_alive > 0) & (n_e_alive > 0)

        # 1. Enemy HP drive
        e_hp_eff = (z_hp * z_alive).sum(axis=-1) + (s_hp * s_alive).sum(axis=-1)
        e_hp_max = n_zealots * ZEALOT_HP_MAX + n_stalkers * STALKER_HP_MAX
        ehc = W_ENEMY_HP * (e_hp_eff / e_hp_max)
        ehc = np.where(n_e_alive == 0, W_WIN_BONUS, ehc)
        comps['enemy_hp'] += ehc

        # 2. Survival
        m_hp_sum = (m_hp * m_alive).sum(axis=-1)
        mm_hp_sum = (mm_hp * mm_alive).sum(axis=-1)
        mv_hp_sum = (mv_hp * mv_alive).sum(axis=-1)
        sv = (
            W_SURV_M * (1.0 - m_hp_sum / max(n_marines * MARINE_HP_MAX, 1e-6))
          + W_SURV_MM * (1.0 - mm_hp_sum / max(n_marauders * MARAUDER_HP_MAX, 1e-6))
          + W_SURV_MV * (1.0 - mv_hp_sum / max(n_medivacs * MEDIVAC_HP_MAX, 1e-6))
          + W_DEAD_M * (n_marines - m_alive.sum(axis=-1))
          + W_DEAD_MM * (n_marauders - mm_alive.sum(axis=-1))
          + W_DEAD_MV * (n_medivacs - n_mv_alive)
        )
        all_dead = (n_bio_alive == 0) & (n_mv_alive == 0)
        sv = sv + np.where(all_dead, W_WIPEOUT, 0.0)
        comps['survival'] += sv

        # Distances (batched)
        # m_pos (N, n_m, 2), z_pos (N, n_z, 2) → (N, n_m, n_z)
        m2z = np.linalg.norm(m_pos[:, :, None, :] - z_pos[:, None, :, :], axis=-1)
        m2s = np.linalg.norm(m_pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1)
        mm2z = np.linalg.norm(mm_pos[:, :, None, :] - z_pos[:, None, :, :], axis=-1)
        mm2s = np.linalg.norm(mm_pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1)

        m2z_eff = np.where(z_alive[:, None, :], m2z, np.inf)
        m2s_eff = np.where(s_alive[:, None, :], m2s, np.inf)
        mm2z_eff = np.where(z_alive[:, None, :], mm2z, np.inf)
        mm2s_eff = np.where(s_alive[:, None, :], mm2s, np.inf)

        m_d_z = np.min(m2z_eff, axis=-1)            # (N, n_m) closest zealot
        m_d_s = np.min(m2s_eff, axis=-1)            # (N, n_m) closest stalker
        mm_d_z = np.min(mm2z_eff, axis=-1)
        mm_d_s = np.min(mm2s_eff, axis=-1)

        # 3. Marine matchup
        # if d_z < 2.5: kite
        # elif n_s > 0: engage stalker
        # else: kite zealot
        kite_z_m = kite_marine_vs_zealot(m_d_z)
        engage_s_m = engage_marine_vs_stalker(m_d_s)
        n_s_alive_b = (n_s_alive > 0)[:, None]
        m_match = np.where(m_d_z < 2.5, kite_z_m,
                  np.where(n_s_alive_b, engage_s_m, kite_z_m))
        m_match = np.where(m_alive, m_match, 0.0)
        m_match_total = W_MATCHUP_M * m_match.sum(axis=-1)
        comps['matchup_marines'] += np.where(active, m_match_total, 0.0)

        # 4. Marauder matchup
        kite_z_mm = kite_marauder_vs_zealot(mm_d_z)
        engage_s_mm = engage_marauder_vs_stalker(mm_d_s)
        mm_match = np.where(mm_d_z < 2.5, kite_z_mm,
                   np.where(n_s_alive_b, engage_s_mm, kite_z_mm))
        mm_match = np.where(mm_alive, mm_match, 0.0)
        mm_match_total = W_MATCHUP_MM * mm_match.sum(axis=-1)
        comps['matchup_marauders'] += np.where(active, mm_match_total, 0.0)

        # 5. Medivac safety
        # mv_pos (N, 1, 2), all enemies (N, n_e, 2)
        # Stack zealots + stalkers; mask by alive
        all_e_pos = np.concatenate([z_pos, s_pos], axis=1)         # (N, n_z+n_s, 2)
        all_e_alive = np.concatenate([z_alive, s_alive], axis=1)   # (N, n_z+n_s)
        mv2e = np.linalg.norm(mv_pos[:, :, None, :] - all_e_pos[:, None, :, :], axis=-1)
        mv2e = np.where(all_e_alive[:, None, :], mv2e, np.inf)
        mv_d_min = np.min(mv2e, axis=-1)                            # (N, n_mv)
        mv_safety_per = medivac_safety(mv_d_min)
        mv_safety_per = np.where(mv_alive, mv_safety_per, 0.0)
        comps['medivac_safety'] += np.where(active,
                                             W_MEDIVAC_SAFETY * mv_safety_per.sum(axis=-1),
                                             0.0)

        # 6. Medivac heal proximity (to most injured bio)
        # Need: per-batch, find most injured alive bio (marine or marauder)
        bio_dmg_marine = (MARINE_HP_MAX - m_hp) * m_alive            # (N, n_m)
        bio_dmg_marauder = (MARAUDER_HP_MAX - mm_hp) * mm_alive      # (N, n_mm)
        # We want the bio unit (across both types) with max damage
        # Concat damages and positions
        bio_pos_all = np.concatenate([m_pos, mm_pos], axis=1)        # (N, n_m+n_mm, 2)
        bio_dmg_all = np.concatenate([bio_dmg_marine, bio_dmg_marauder], axis=-1)
        # Mask dead with -inf so they're never picked
        bio_alive_all = np.concatenate([m_alive, mm_alive], axis=-1)
        bio_dmg_masked = np.where(bio_alive_all, bio_dmg_all, -np.inf)
        most_injured_idx = np.argmax(bio_dmg_masked, axis=-1)        # (N,)
        # Gather position of most-injured bio
        arange_N = np.arange(N)
        most_inj_pos = bio_pos_all[arange_N, most_injured_idx]       # (N, 2)
        most_inj_dmg = bio_dmg_all[arange_N, most_injured_idx]       # (N,)
        # Distance from medivac to most injured
        d_mv_inj = np.linalg.norm(mv_pos[:, 0] - most_inj_pos, axis=-1)  # (N,)
        # Only fire heal_proximity reward when there's >1 HP of damage
        any_injury = most_inj_dmg > 1.0
        heal_per = medivac_heal_proximity(d_mv_inj)
        heal_per = np.where(any_injury & mv_alive[:, 0], heal_per, 0.0)
        comps['medivac_heal'] += np.where(active, W_MEDIVAC_HEAL * heal_per, 0.0)

        # 7. Focus fire on lowest-HP alive enemy
        all_e_hp = np.concatenate([z_hp, s_hp], axis=-1)
        all_e_hp_masked = np.where(all_e_alive, all_e_hp, np.inf)
        weakest_idx = np.argmin(all_e_hp_masked, axis=-1)            # (N,)
        weakest_pos = all_e_pos[arange_N, weakest_idx]               # (N, 2)
        weakest_alive = all_e_alive[arange_N, weakest_idx]
        # Count bio in their range of weakest
        d_m_w = np.linalg.norm(m_pos - weakest_pos[:, None, :], axis=-1)   # (N, n_m)
        d_mm_w = np.linalg.norm(mm_pos - weakest_pos[:, None, :], axis=-1)
        m_in_range_of_w = (d_m_w <= MARINE_RANGE) & m_alive
        mm_in_range_of_w = (d_mm_w <= MARAUDER_RANGE) & mm_alive
        n_in_range = m_in_range_of_w.sum(axis=-1) + mm_in_range_of_w.sum(axis=-1)
        ff = -W_FOCUS_FIRE * n_in_range
        ff = np.where(weakest_alive, ff, 0.0)
        comps['focus_fire'] += np.where(active, ff, 0.0)

        # 8. Stalker priority — penalize zero bio in MARAUDER_RANGE of each alive stalker
        # For each stalker: min distance from any alive bio
        bio_pos = bio_pos_all                                         # (N, n_m+n_mm, 2)
        bio_alive = bio_alive_all
        bio_to_s = np.linalg.norm(bio_pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1)
        bio_to_s_eff = np.where(bio_alive[:, :, None], bio_to_s, np.inf)
        s_min_d = np.min(bio_to_s_eff, axis=1)                        # (N, n_s)
        sp_per = np.where(s_alive & (s_min_d > MARAUDER_RANGE),
                          W_STALKER_PRIORITY * (s_min_d - MARAUDER_RANGE), 0.0)
        comps['stalker_priority'] += np.where(active, sp_per.sum(axis=-1), 0.0)

        # 9. DPS uptime
        # per marine: in range of any enemy AND weapon ready
        m_in_range_any = (np.min(np.concatenate([m2z_eff, m2s_eff], axis=-1), axis=-1)
                          <= MARINE_RANGE)
        mm_in_range_any = (np.min(np.concatenate([mm2z_eff, mm2s_eff], axis=-1), axis=-1)
                           <= MARAUDER_RANGE)
        m_dps = (m_alive & m_ready & m_in_range_any) * -W_DPS_M
        mm_dps = (mm_alive & mm_ready & mm_in_range_any) * -W_DPS_MM
        comps['dps_uptime'] += np.where(active,
                                         m_dps.sum(axis=-1) + mm_dps.sum(axis=-1),
                                         0.0)

        # Cohesion (anti-split). Max pairwise dist between alive bio.
        if W_COHESION > 0.0:
            bio_pos_t = np.concatenate([m_pos, mm_pos], axis=1)             # (N, n_bio, 2)
            bio_alive_t = np.concatenate([m_alive, mm_alive], axis=-1)
            d_pair = np.linalg.norm(bio_pos_t[:, :, None, :]
                                     - bio_pos_t[:, None, :, :], axis=-1)
            both_alive = bio_alive_t[:, :, None] & bio_alive_t[:, None, :]
            d_pair = np.where(both_alive, d_pair, -np.inf)
            max_pair = np.max(d_pair, axis=(1, 2))                          # (N,)
            n_alive_bio = bio_alive_t.sum(axis=-1)
            coh = np.where(n_alive_bio >= 2,
                           W_COHESION * np.maximum(0.0, max_pair - COHESION_THRESHOLD),
                           0.0)
            comps.setdefault('cohesion', np.zeros(N))
            comps['cohesion'] += np.where(active, coh, 0.0)

        # Optional structural extension (variants set EXTRA_COST_FN_BATCH)
        if EXTRA_COST_FN_BATCH is not None:
            ctx = dict(
                m_pos=m_pos, mm_pos=mm_pos, mv_pos=mv_pos,
                z_pos=z_pos, s_pos=s_pos,
                m_alive=m_alive, mm_alive=mm_alive,
                z_alive=z_alive, s_alive=s_alive,
                active=active, arange_N=np.arange(N),
            )
            for k, v in EXTRA_COST_FN_BATCH(ctx).items():
                if k not in comps:
                    comps[k] = np.zeros(N)
                comps[k] += v

    err_ctx.__exit__(None, None, None)
    total = sum(comps.values())
    return total, comps
