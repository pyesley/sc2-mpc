"""
Vectorized SMPC for Bio + Medivac vs Mixed Protoss.

Composition (fixed at scenario): 6 Marines + 2 Marauders + 1 Medivac
                                  vs 3 Zealots + 2 Stalkers.

Action space per step is (n_m + n_mm + n_mv) × 2 = 18-D. Per-unit-type
sampling (different distance bands for each role) keeps candidates in
plausible micro behaviors. Cost is selected via COST_MODULE env var
(defaults to cost_bio).
"""

import os
import importlib
import numpy as np
from typing import Dict, List


# Cost module via env var (Eureka harness compat)
COST_MODULE_NAME = os.environ.get("COST_MODULE", "cost_bio")
_cost_mod = importlib.import_module(COST_MODULE_NAME)
compute_cost_batch = _cost_mod.compute_cost_batch


# ─── Game constants ──────────────────────────────────────────
MARINE_SPEED = 2.25
MARAUDER_SPEED = 2.25
MEDIVAC_SPEED = 2.5
ZEALOT_SPEED = 2.25
STALKER_SPEED = 2.95

MARINE_RANGE = 5.0
MARAUDER_RANGE = 6.0
STALKER_RANGE = 6.0
MEDIVAC_HEAL_RANGE = 4.0

MARINE_DPS = 9.8
MARAUDER_DPS = 9.3
STALKER_DPS = 9.7
ZEALOT_DPS = 26.3

MEDIVAC_HEAL_RATE = 9.0     # HP / s

MARINE_HP_MAX = 45.0
MARAUDER_HP_MAX = 125.0
MEDIVAC_HP_MAX = 150.0
ZEALOT_HP_MAX = 150.0
STALKER_HP_MAX = 160.0

ZEALOT_SWITCH_SHARPNESS = 3.0
PURSUIT_NOISE = 0.15
EXEC_NOISE = 0.05


# ─── Action sampling ─────────────────────────────────────────
def _sample_kiting_unit(N, n_units, alive, my_pos, threat_pos,
                          d_threat, range_band_thresholds, weights_table):
    """Generic kiting sampler.

    range_band_thresholds: sorted list e.g. [1.8, 3.5, 5.0, 7.0]
    weights_table: per-band {'w_away_lo','w_away_hi','w_tang_lo','w_tang_hi','hold_p'}
    Returns (N, n_units, 2) directions.
    """
    # away vector and tangent per (N, n_units)
    away = my_pos - threat_pos                                   # (N, n, 2)
    away_norm = np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 0.1)
    away_n = away / away_norm
    tangent = np.stack([-away_n[..., 1], away_n[..., 0]], axis=-1)
    sign = np.where(np.random.random((N, n_units)) < 0.5, 1.0, -1.0)
    tang = tangent * sign[..., None]

    w_away = np.zeros((N, n_units))
    w_tang = np.zeros((N, n_units))
    hold = np.zeros((N, n_units), dtype=bool)

    bands = np.zeros((N, n_units), dtype=int)
    for i, thresh in enumerate(range_band_thresholds):
        bands += (d_threat >= thresh).astype(int)

    for band_i, w in enumerate(weights_table):
        m = (bands == band_i)
        if w.get('hold_p', 0) > 0:
            r = np.random.random((N, n_units))
            hold |= m & (r < w['hold_p'])
        w_away = np.where(m, np.random.uniform(w['w_away_lo'], w['w_away_hi'], (N, n_units)), w_away)
        w_tang = np.where(m, np.random.uniform(w['w_tang_lo'], w['w_tang_hi'], (N, n_units)), w_tang)

    dirs = w_away[..., None] * away_n + w_tang[..., None] * tang \
           + 0.08 * np.random.randn(N, n_units, 2)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = np.where(norms > 0.1, dirs / np.maximum(norms, 1e-6), 0.0)
    dirs = np.where(hold[..., None], 0.0, dirs)
    dirs = np.where(alive[..., None], dirs, 0.0)
    return dirs


# Distance-band weight tables (same shape as scenario_circle's sampler)
MARINE_WEIGHTS = [
    dict(w_away_lo=0.7, w_away_hi=1.0, w_tang_lo=-0.4, w_tang_hi=0.4),                  # d < 1.8
    dict(w_away_lo=-0.1, w_away_hi=0.4, w_tang_lo=0.5, w_tang_hi=1.0),                  # 1.8-3.5
    dict(w_away_lo=-0.3, w_away_hi=0.3, w_tang_lo=-0.5, w_tang_hi=0.5, hold_p=0.5),     # 3.5-5
    dict(w_away_lo=-0.7, w_away_hi=-0.2, w_tang_lo=-0.4, w_tang_hi=0.4, hold_p=0.3),    # 5-7
    dict(w_away_lo=-1.0, w_away_hi=-0.5, w_tang_lo=-0.3, w_tang_hi=0.3),                # 7+
]
MARAUDER_WEIGHTS = [
    dict(w_away_lo=0.7, w_away_hi=1.0, w_tang_lo=-0.4, w_tang_hi=0.4),                  # d < 2
    dict(w_away_lo=-0.1, w_away_hi=0.4, w_tang_lo=0.4, w_tang_hi=0.9),                  # 2-4
    dict(w_away_lo=-0.3, w_away_hi=0.3, w_tang_lo=-0.5, w_tang_hi=0.5, hold_p=0.55),    # 4-6
    dict(w_away_lo=-0.7, w_away_hi=-0.2, w_tang_lo=-0.4, w_tang_hi=0.4, hold_p=0.3),    # 6-8
    dict(w_away_lo=-1.0, w_away_hi=-0.4, w_tang_lo=-0.3, w_tang_hi=0.3),                # 8+
]


def sample_actions_batch(state, N, horizon):
    """Sample N candidate action sequences for all controlled units.
    Returns (N, n_total_units, horizon, 2) where the unit ordering is
    [marines..., marauders..., medivac]."""
    n_m = state.n_marines
    n_mm = state.n_marauders
    n_mv = state.n_medivacs
    n_total = n_m + n_mm + n_mv

    m_pos0 = np.asarray(state.marine_positions, dtype=np.float64)
    mm_pos0 = np.asarray(state.marauder_positions, dtype=np.float64)
    mv_pos0 = np.asarray(state.medivac_positions, dtype=np.float64)
    z_pos0 = np.asarray(state.zealot_positions, dtype=np.float64)
    s_pos0 = np.asarray(state.stalker_positions, dtype=np.float64)
    z_alive0 = np.asarray(state.zealot_alive, dtype=bool)
    s_alive0 = np.asarray(state.stalker_alive, dtype=bool)
    m_alive0 = np.asarray(state.marine_hps, dtype=np.float64) > 0
    mm_alive0 = np.asarray(state.marauder_hps, dtype=np.float64) > 0
    mv_alive0 = np.asarray(state.medivac_hps, dtype=np.float64) > 0

    # Closest enemy per marine / marauder (zealot or stalker)
    all_enemy_pos = np.concatenate([z_pos0, s_pos0], axis=0)             # (n_z+n_s, 2)
    all_enemy_alive = np.concatenate([z_alive0, s_alive0], axis=0)

    def _closest(my_pos):
        d = np.linalg.norm(my_pos[:, None, :] - all_enemy_pos[None, :, :], axis=-1)
        d = np.where(all_enemy_alive[None, :], d, np.inf)
        idx = np.argmin(d, axis=-1)
        return d.min(axis=-1), all_enemy_pos[idx]

    m_d_threat, m_threat_pos = _closest(m_pos0)
    mm_d_threat, mm_threat_pos = _closest(mm_pos0)

    # Broadcast initial state into (N, n, ...)
    m_pos_b = np.broadcast_to(m_pos0[None], (N, n_m, 2))
    mm_pos_b = np.broadcast_to(mm_pos0[None], (N, n_mm, 2))
    m_threat_b = np.broadcast_to(m_threat_pos[None], (N, n_m, 2))
    mm_threat_b = np.broadcast_to(mm_threat_pos[None], (N, n_mm, 2))
    m_d_b = np.broadcast_to(m_d_threat[None], (N, n_m))
    mm_d_b = np.broadcast_to(mm_d_threat[None], (N, n_mm))
    m_alive_b = np.broadcast_to(m_alive0[None], (N, n_m))
    mm_alive_b = np.broadcast_to(mm_alive0[None], (N, n_mm))
    mv_alive_b = np.broadcast_to(mv_alive0[None], (N, n_mv))

    # Medivac: needs target position = injured bio centroid (or just bio centroid)
    # For sampling we approximate: medivac biases toward bio centroid, away from stalkers
    bio_pos_init = np.concatenate([m_pos0[m_alive0], mm_pos0[mm_alive0]], axis=0) \
                   if (m_alive0.any() or mm_alive0.any()) \
                   else m_pos0[:1]
    bio_centroid = bio_pos_init.mean(axis=0) if len(bio_pos_init) > 0 else mv_pos0[0]
    # Closest stalker to medivac
    if s_alive0.any():
        d_mv_s = np.linalg.norm(mv_pos0[0] - s_pos0[s_alive0], axis=-1)
        nearest_stalker = s_pos0[s_alive0][int(np.argmin(d_mv_s))]
        mv_threat_pos_init = nearest_stalker
        mv_d_threat_init = float(d_mv_s.min())
    else:
        mv_threat_pos_init = mv_pos0[0] + np.array([100.0, 0.0])
        mv_d_threat_init = 100.0

    actions = np.zeros((N, n_total, horizon, 2), dtype=np.float64)

    for h in range(horizon):
        # Marines: kite-based on closest enemy
        m_dirs = _sample_kiting_unit(N, n_m, m_alive_b, m_pos_b, m_threat_b,
                                       m_d_b, [1.8, 3.5, 5.0, 7.0], MARINE_WEIGHTS)
        # Marauders
        mm_dirs = _sample_kiting_unit(N, n_mm, mm_alive_b, mm_pos_b, mm_threat_b,
                                        mm_d_b, [2.0, 4.0, 6.0, 8.0], MARAUDER_WEIGHTS)
        # Medivac: bias toward bio centroid, away from nearest stalker
        # Simple sampler: 60% noise around centroid direction, 40% retreat from stalker
        mv_dirs = np.zeros((N, n_mv, 2))
        to_centroid = bio_centroid - mv_pos0[0]
        d_centroid = np.linalg.norm(to_centroid)
        if d_centroid > 0.1:
            to_centroid_n = to_centroid / d_centroid
        else:
            to_centroid_n = np.array([0.0, 0.0])
        away_stalker = mv_pos0[0] - mv_threat_pos_init
        d_aws = np.linalg.norm(away_stalker)
        if d_aws > 0.1:
            away_stalker_n = away_stalker / d_aws
        else:
            away_stalker_n = np.array([1.0, 0.0])
        # Per candidate weighted blend
        u_blend = np.random.uniform(0.0, 1.0, N)
        if mv_d_threat_init < STALKER_RANGE + 1.0:
            # In stalker threat range — heavy retreat bias
            w_retreat = np.random.uniform(0.5, 1.0, N)
            w_centroid = np.random.uniform(-0.2, 0.4, N)
        else:
            w_retreat = np.random.uniform(-0.2, 0.3, N)
            w_centroid = np.random.uniform(0.4, 1.0, N)
        mv_d = (w_retreat[:, None] * away_stalker_n[None, :]
                + w_centroid[:, None] * to_centroid_n[None, :]
                + 0.1 * np.random.randn(N, 2))
        mv_norms = np.linalg.norm(mv_d, axis=-1, keepdims=True)
        mv_d = np.where(mv_norms > 0.1, mv_d / np.maximum(mv_norms, 1e-6), 0.0)
        mv_d = np.where(mv_alive_b[:, 0:1, None], mv_d[:, None, :], 0.0)
        mv_dirs = mv_d.reshape(N, n_mv, 2) if mv_d.ndim == 3 and mv_d.shape[1] == 1 else mv_d

        # Pack into combined actions tensor [marines..., marauders..., medivac]
        actions[:, :n_m, h, :] = m_dirs
        actions[:, n_m:n_m + n_mm, h, :] = mm_dirs
        actions[:, n_m + n_mm:, h, :] = mv_dirs

    return actions


# ─── Simulator ───────────────────────────────────────────────
def simulate_batch(state, actions, dt=0.4, stochastic=True):
    """Roll out trajectories. actions (N, n_total, H, 2) where unit
    ordering is [marines..., marauders..., medivac].

    Returns dict of (N, H, ...) arrays."""
    N, n_total, H, _ = actions.shape
    n_m = state.n_marines
    n_mm = state.n_marauders
    n_mv = state.n_medivacs
    n_z = state.n_zealots
    n_s = state.n_stalkers

    # Slice into per-type action streams
    m_act = actions[:, :n_m, :, :]
    mm_act = actions[:, n_m:n_m + n_mm, :, :]
    mv_act = actions[:, n_m + n_mm:, :, :]

    # Init state per-batch
    m_pos = np.tile(np.asarray(state.marine_positions)[None], (N, 1, 1))
    mm_pos = np.tile(np.asarray(state.marauder_positions)[None], (N, 1, 1))
    mv_pos = np.tile(np.asarray(state.medivac_positions)[None], (N, 1, 1))
    z_pos = np.tile(np.asarray(state.zealot_positions)[None], (N, 1, 1))
    s_pos = np.tile(np.asarray(state.stalker_positions)[None], (N, 1, 1))
    m_hp = np.tile(np.asarray(state.marine_hps)[None], (N, 1))
    mm_hp = np.tile(np.asarray(state.marauder_hps)[None], (N, 1))
    mv_hp = np.tile(np.asarray(state.medivac_hps)[None], (N, 1))
    z_hp = np.tile(np.asarray(state.zealot_hps)[None], (N, 1))
    s_hp = np.tile(np.asarray(state.stalker_hps)[None], (N, 1))
    z_alive = np.tile(np.asarray(state.zealot_alive)[None], (N, 1))
    s_alive = np.tile(np.asarray(state.stalker_alive)[None], (N, 1))

    # Output buffers
    out_m_pos = np.zeros((N, H, n_m, 2))
    out_mm_pos = np.zeros((N, H, n_mm, 2))
    out_mv_pos = np.zeros((N, H, n_mv, 2))
    out_z_pos = np.zeros((N, H, n_z, 2))
    out_s_pos = np.zeros((N, H, n_s, 2))
    out_m_hp = np.zeros((N, H, n_m))
    out_mm_hp = np.zeros((N, H, n_mm))
    out_mv_hp = np.zeros((N, H, n_mv))
    out_z_hp = np.zeros((N, H, n_z))
    out_s_hp = np.zeros((N, H, n_s))
    out_z_alive = np.zeros((N, H, n_z), dtype=bool)
    out_s_alive = np.zeros((N, H, n_s), dtype=bool)
    out_m_ready = np.zeros((N, H, n_m), dtype=bool)
    out_mm_ready = np.zeros((N, H, n_mm), dtype=bool)

    arange_N = np.arange(N)

    for t in range(H):
        m_a = m_act[:, :, t, :]
        mm_a = mm_act[:, :, t, :]
        mv_a = mv_act[:, :, t, :]

        m_alive = m_hp > 0
        mm_alive = mm_hp > 0
        mv_alive = mv_hp > 0

        m_moving = np.linalg.norm(m_a, axis=-1) > 0.1
        mm_moving = np.linalg.norm(mm_a, axis=-1) > 0.1

        # Move (with execution noise)
        if stochastic:
            m_pos = m_pos + m_a * MARINE_SPEED * dt + np.random.normal(0, EXEC_NOISE, (N, n_m, 2))
            mm_pos = mm_pos + mm_a * MARAUDER_SPEED * dt + np.random.normal(0, EXEC_NOISE, (N, n_mm, 2))
            mv_pos = mv_pos + mv_a * MEDIVAC_SPEED * dt + np.random.normal(0, EXEC_NOISE, (N, n_mv, 2))
        else:
            m_pos = m_pos + m_a * MARINE_SPEED * dt
            mm_pos = mm_pos + mm_a * MARAUDER_SPEED * dt
            mv_pos = mv_pos + mv_a * MEDIVAC_SPEED * dt

        # Bio positions stacked for enemy targeting (n_m + n_mm)
        bio_pos = np.concatenate([m_pos, mm_pos], axis=1)        # (N, n_m+n_mm, 2)
        bio_alive = np.concatenate([m_alive, mm_alive], axis=-1)
        n_bio = bio_pos.shape[1]

        # ── Zealots: each chases closest alive bio, melees if in range ──
        z2bio = np.linalg.norm(z_pos[:, :, None, :] - bio_pos[:, None, :, :], axis=-1)  # (N, n_z, n_bio)
        z2bio_eff = np.where(bio_alive[:, None, :], z2bio, np.inf)
        if stochastic:
            logits = -ZEALOT_SWITCH_SHARPNESS * z2bio_eff
            gumbels = -np.log(-np.log(np.random.uniform(1e-12, 1, (N, n_z, n_bio))))
            z_target = np.argmax(logits + gumbels, axis=-1)            # (N, n_z)
        else:
            z_target = np.argmin(z2bio_eff, axis=-1)
        z_target_pos = bio_pos[arange_N[:, None], z_target]            # (N, n_z, 2)
        zr = z_target_pos - z_pos
        zd = np.linalg.norm(zr, axis=-1, keepdims=True)
        zdir = np.where(zd > 0.1, zr / np.maximum(zd, 1e-6), 0.0)
        if stochastic:
            ang = np.random.normal(0, PURSUIT_NOISE, (N, n_z))
            ca = np.cos(ang)
            sa = np.sin(ang)
            zdir = np.stack([
                ca * zdir[..., 0] - sa * zdir[..., 1],
                sa * zdir[..., 0] + ca * zdir[..., 1],
            ], axis=-1)
        z_step = zdir * np.minimum(ZEALOT_SPEED * dt, zd[..., 0])[..., None]
        z_pos = z_pos + z_step * z_alive[..., None]

        # Zealot melee damage (if in range 1.0)
        new_zd = np.linalg.norm(z_target_pos - z_pos, axis=-1)
        z_in_melee = (new_zd < 1.0) & z_alive
        z_dmg = z_in_melee * ZEALOT_DPS * dt
        # bio target index (N, n_z), scatter -damage onto bio HP
        # bio HP = stacked [m_hp ; mm_hp] — we need to split back
        bio_hp = np.concatenate([m_hp, mm_hp], axis=-1)
        z_b = np.broadcast_to(arange_N[:, None], (N, n_z))
        np.add.at(bio_hp, (z_b, z_target), -z_dmg)
        m_hp = bio_hp[:, :n_m]
        mm_hp = bio_hp[:, n_m:]

        # ── Stalkers: chase closest bio to get in range, then hold and shoot ──
        s2bio = np.linalg.norm(s_pos[:, :, None, :] - bio_pos[:, None, :, :], axis=-1)
        s2bio_eff = np.where(bio_alive[:, None, :], s2bio, np.inf)
        s_target = np.argmin(s2bio_eff, axis=-1)                       # (N, n_s)
        s_target_pos = bio_pos[arange_N[:, None], s_target]
        sr = s_target_pos - s_pos
        sd = np.linalg.norm(sr, axis=-1, keepdims=True)
        s_dir = np.where(sd > 0.1, sr / np.maximum(sd, 1e-6), 0.0)
        # Move only if outside range
        out_of_range = (sd[..., 0] > STALKER_RANGE) & s_alive
        step = s_dir * np.minimum(STALKER_SPEED * dt, sd[..., 0] - STALKER_RANGE + 0.5)[..., None]
        s_pos = s_pos + step * out_of_range[..., None]
        # Stalker shoots if in range
        s_in_range = (sd[..., 0] <= STALKER_RANGE) & s_alive
        s_dmg = s_in_range * STALKER_DPS * dt
        s_b = np.broadcast_to(arange_N[:, None], (N, n_s))
        np.add.at(bio_hp, (s_b, s_target), -s_dmg)
        m_hp = bio_hp[:, :n_m]
        mm_hp = bio_hp[:, n_m:]

        # ── Marines/marauders shoot closest alive enemy in range, if stationary ──
        all_e_pos = np.concatenate([z_pos, s_pos], axis=1)             # (N, n_z+n_s, 2)
        all_e_alive = np.concatenate([z_alive, s_alive], axis=1)
        n_e = all_e_pos.shape[1]

        # Marines
        m2e = np.linalg.norm(m_pos[:, :, None, :] - all_e_pos[:, None, :, :], axis=-1)  # (N, n_m, n_e)
        m2e_eff = np.where(all_e_alive[:, None, :], m2e, np.inf)
        m_target = np.argmin(m2e_eff, axis=-1)
        m_target_d = np.min(m2e_eff, axis=-1)
        m_shoots = m_alive & ~m_moving & (m_target_d <= MARINE_RANGE)
        m_dmg = m_shoots * MARINE_DPS * dt
        # Damage scatter into combined enemy HP array
        e_hp = np.concatenate([z_hp, s_hp], axis=-1)
        m_b = np.broadcast_to(arange_N[:, None], (N, n_m))
        np.add.at(e_hp, (m_b, m_target), -m_dmg)
        z_hp = e_hp[:, :n_z]
        s_hp = e_hp[:, n_z:]

        # Marauders
        mm2e = np.linalg.norm(mm_pos[:, :, None, :] - all_e_pos[:, None, :, :], axis=-1)
        mm2e_eff = np.where(all_e_alive[:, None, :], mm2e, np.inf)
        mm_target = np.argmin(mm2e_eff, axis=-1)
        mm_target_d = np.min(mm2e_eff, axis=-1)
        mm_shoots = mm_alive & ~mm_moving & (mm_target_d <= MARAUDER_RANGE)
        mm_dmg = mm_shoots * MARAUDER_DPS * dt
        e_hp = np.concatenate([z_hp, s_hp], axis=-1)
        mm_b = np.broadcast_to(arange_N[:, None], (N, n_mm))
        np.add.at(e_hp, (mm_b, mm_target), -mm_dmg)
        z_hp = e_hp[:, :n_z]
        s_hp = e_hp[:, n_z:]

        # ── Medivac heal: most-injured bio in heal range gets +9 HP/s ──
        # Simplification: assume infinite energy.
        if n_mv > 0:
            mv2bio = np.linalg.norm(mv_pos[:, :, None, :] - bio_pos[:, None, :, :], axis=-1)
            # Most injured = max damage = (max_hp - hp). Need per-bio max_hp.
            bio_hp_max = np.concatenate([
                np.full((N, n_m), MARINE_HP_MAX),
                np.full((N, n_mm), MARAUDER_HP_MAX),
            ], axis=-1)
            bio_dmg = (bio_hp_max - bio_hp) * bio_alive    # (N, n_bio)
            # Mask: only injured bios within heal range of the medivac (assume 1 mv)
            in_range = (mv2bio[:, 0, :] <= MEDIVAC_HEAL_RANGE) & bio_alive
            bio_dmg_in_range = np.where(in_range, bio_dmg, -1.0)
            heal_target = np.argmax(bio_dmg_in_range, axis=-1)         # (N,)
            heal_target_dmg = bio_dmg_in_range[arange_N, heal_target]
            heal_active = (heal_target_dmg > 0.1) & mv_alive[:, 0]
            heal_amount = heal_active * MEDIVAC_HEAL_RATE * dt
            np.add.at(bio_hp, (arange_N, heal_target), heal_amount)
            # cap at max
            bio_hp = np.minimum(bio_hp, bio_hp_max)
            m_hp = bio_hp[:, :n_m]
            mm_hp = bio_hp[:, n_m:]

        # ── Death checks ──
        z_alive = z_alive & (z_hp > 0)
        s_alive = s_alive & (s_hp > 0)
        z_hp = np.maximum(z_hp, 0)
        s_hp = np.maximum(s_hp, 0)
        m_hp = np.maximum(m_hp, 0)
        mm_hp = np.maximum(mm_hp, 0)
        mv_hp = np.maximum(mv_hp, 0)

        out_m_pos[:, t] = m_pos
        out_mm_pos[:, t] = mm_pos
        out_mv_pos[:, t] = mv_pos
        out_z_pos[:, t] = z_pos
        out_s_pos[:, t] = s_pos
        out_m_hp[:, t] = m_hp
        out_mm_hp[:, t] = mm_hp
        out_mv_hp[:, t] = mv_hp
        out_z_hp[:, t] = z_hp
        out_s_hp[:, t] = s_hp
        out_z_alive[:, t] = z_alive
        out_s_alive[:, t] = s_alive
        out_m_ready[:, t] = ~m_moving
        out_mm_ready[:, t] = ~mm_moving

    return {
        'm_pos': out_m_pos, 'mm_pos': out_mm_pos, 'mv_pos': out_mv_pos,
        'z_pos': out_z_pos, 's_pos': out_s_pos,
        'm_hp': out_m_hp, 'mm_hp': out_mm_hp, 'mv_hp': out_mv_hp,
        'z_hp': out_z_hp, 's_hp': out_s_hp,
        'z_alive': out_z_alive, 's_alive': out_s_alive,
        'm_ready': out_m_ready, 'mm_ready': out_mm_ready,
    }


def mpc_select_action_vectorized(state, n_candidates=64, n_scenarios=4,
                                  horizon=8, dt=0.4, cvar_alpha=0.3):
    """Returns (actions_list, components) where actions_list is one
    np.ndarray(2) per controlled unit in order [marines..., marauders..., medivac]."""
    n_m = state.n_marines
    n_mm = state.n_marauders
    n_mv = state.n_medivacs
    n_z = state.n_zealots
    n_s = state.n_stalkers
    n_total = n_m + n_mm + n_mv

    C = n_candidates
    S = n_scenarios

    actions_c = sample_actions_batch(state, C, horizon)               # (C, n_total, H, 2)
    actions = np.repeat(actions_c, S, axis=0)                         # (C*S, ...)

    traj = simulate_batch(state, actions, dt=dt, stochastic=True)

    total, comps = compute_cost_batch(traj, n_m, n_mm, n_mv, n_z, n_s)

    total_2d = total.reshape(C, S)
    sorted_costs = np.sort(total_2d, axis=1)[:, ::-1]
    n_tail = max(1, int(np.ceil(S * cvar_alpha)))
    cvar = np.mean(sorted_costs[:, :n_tail], axis=1)

    best_idx = int(np.argmin(cvar))
    best_actions = [actions_c[best_idx, u, 0, :].copy() for u in range(n_total)]

    comps_2d = {k: v.reshape(C, S) for k, v in comps.items()}
    sorted_idx = np.argsort(total_2d[best_idx])
    median_idx = int(sorted_idx[S // 2])
    best_components = {k: float(comps_2d[k][best_idx, median_idx]) for k in comps_2d}

    return best_actions, best_components
