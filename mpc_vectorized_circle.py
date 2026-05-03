"""
Vectorized SMPC for Round 29 (3 Marines vs 2 Zealots).

Mirrors mpc_vectorized.py's pattern but for the circle scenario's state
shape (n_marines=3, n_zealots=2). All n_candidates × n_scenarios
trajectories are simulated in one batched numpy pass.

Spec parity: scenario_circle.simulate_circle / sample_action_circle
remain the single-state reference implementations. The functions here
must reproduce their behavior in batched form.
"""

import os
import importlib
import numpy as np
from typing import Dict, List, Tuple

# Cost module is selectable via env var (matches scenario_circle.py).
COST_MODULE_NAME = os.environ.get("COST_MODULE", "cost_circle")
_cost_mod = importlib.import_module(COST_MODULE_NAME)
compute_cost_batch = _cost_mod.compute_cost_batch


# ─── Game constants (must match scenario_circle.py) ──────────
MARINE_RANGE = 5.0
MARINE_SPEED = 2.25
ZEALOT_SPEED = 2.25
MARINE_DPS = 9.8
ZEALOT_DPS = 26.3

ZEALOT_SWITCH_SHARPNESS = 3.0
ZEALOT_PURSUIT_NOISE = 0.15
MARINE_EXEC_NOISE = 0.05


def sample_actions_batch(state, N, horizon):
    """Sample N candidate (n_marines, horizon, 2) action sequences.

    Distance-based per marine, identical sampling logic to
    scenario_circle.sample_action_circle but batched. The closest-zealot
    distance per marine is computed from the INITIAL state and held
    fixed across the horizon (same simplification as the loop version).
    """
    n_m = state.n_marines
    n_z = state.n_zealots

    m_pos0 = np.asarray(state.marine_positions, dtype=np.float64)   # (n_m, 2)
    z_pos0 = np.asarray(state.zealot_positions, dtype=np.float64)   # (n_z, 2)
    z_alive0 = np.asarray(state.zealot_alive, dtype=bool)           # (n_z,)

    # Closest live zealot per marine
    diffs = m_pos0[:, None, :] - z_pos0[None, :, :]                 # (n_m, n_z, 2)
    dists = np.linalg.norm(diffs, axis=-1)                          # (n_m, n_z)
    dists = np.where(z_alive0[None, :], dists, np.inf)
    closest_idx = np.argmin(dists, axis=-1)                         # (n_m,)
    d_near = np.min(dists, axis=-1)                                 # (n_m,)
    z_near = z_pos0[closest_idx]                                    # (n_m, 2)

    away = m_pos0 - z_near                                          # (n_m, 2)
    away_norm = np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 0.1)
    away_n = away / away_norm
    tangent = np.stack([-away_n[:, 1], away_n[:, 0]], axis=-1)      # (n_m, 2)

    # Broadcast per-marine constants over batch
    d_b = np.broadcast_to(d_near[None, :], (N, n_m))                # (N, n_m)
    away_b = np.broadcast_to(away_n[None, :, :], (N, n_m, 2))
    tang_b = np.broadcast_to(tangent[None, :, :], (N, n_m, 2))

    actions = np.zeros((N, n_m, horizon, 2), dtype=np.float64)

    for h in range(horizon):
        # Random tangent sign per (N, n_m)
        sign = np.where(np.random.random((N, n_m)) < 0.5, 1.0, -1.0)
        tang = tang_b * sign[:, :, None]                            # (N, n_m, 2)

        w_away = np.zeros((N, n_m))
        w_tang = np.zeros((N, n_m))
        hold = np.zeros((N, n_m), dtype=bool)

        # Flee band  d < 1.8
        m_flee = d_b < 1.8
        w_away = np.where(m_flee, np.random.uniform(0.7, 1.0, (N, n_m)), w_away)
        w_tang = np.where(m_flee, np.random.uniform(-0.4, 0.4, (N, n_m)), w_tang)

        # Kite band  1.8 <= d < 3.5
        m_kite = (d_b >= 1.8) & (d_b < 3.5)
        w_away = np.where(m_kite, np.random.uniform(-0.1, 0.4, (N, n_m)), w_away)
        w_tang = np.where(m_kite, np.random.uniform(0.5, 1.0, (N, n_m)), w_tang)

        # In-range band  3.5 <= d <= 5.0  (55% hold, 25% noise, 20% slight)
        m_in = (d_b >= 3.5) & (d_b <= MARINE_RANGE)
        r = np.random.random((N, n_m))
        m_in_hold = m_in & (r < 0.55)
        m_in_noise = m_in & (r >= 0.55) & (r < 0.8)
        m_in_slight = m_in & (r >= 0.8)
        hold |= m_in_hold
        w_away = np.where(m_in_noise, np.random.uniform(-0.3, 0.3, (N, n_m)), w_away)
        w_tang = np.where(m_in_noise, np.random.uniform(-0.5, 0.5, (N, n_m)), w_tang)
        w_away = np.where(m_in_slight, np.random.uniform(-0.2, 0.5, (N, n_m)), w_away)
        w_tang = np.where(m_in_slight, np.random.uniform(-0.4, 0.4, (N, n_m)), w_tang)

        # Almost-range band  5.0 < d < 7.0  (35% hold, 65% close in)
        m_close = (d_b > MARINE_RANGE) & (d_b < 7.0)
        r2 = np.random.random((N, n_m))
        m_cl_hold = m_close & (r2 < 0.35)
        m_cl_close = m_close & (r2 >= 0.35)
        hold |= m_cl_hold
        w_away = np.where(m_cl_close, np.random.uniform(-0.7, -0.2, (N, n_m)), w_away)
        w_tang = np.where(m_cl_close, np.random.uniform(-0.4, 0.4, (N, n_m)), w_tang)

        # Very-far band  d >= 7.0
        m_far = d_b >= 7.0
        w_away = np.where(m_far, np.random.uniform(-1.0, -0.5, (N, n_m)), w_away)
        w_tang = np.where(m_far, np.random.uniform(-0.3, 0.3, (N, n_m)), w_tang)

        # Compose direction
        dirs = w_away[..., None] * away_b + w_tang[..., None] * tang
        dirs = dirs + 0.08 * np.random.randn(N, n_m, 2)

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = np.where(norms > 0.1, dirs / np.maximum(norms, 1e-6), 0.0)
        dirs = np.where(hold[..., None], 0.0, dirs)

        actions[:, :, h, :] = dirs

    return actions


def simulate_batch(state, actions, dt=0.4, stochastic=True):
    """Roll out N action sequences forward in time.

    actions: (N, n_m, H, 2)

    Returns dict of (N, H, ...) trajectory arrays.
    """
    N, n_m, H, _ = actions.shape
    n_z = state.n_zealots

    m_pos = np.tile(np.asarray(state.marine_positions, dtype=np.float64)[None],
                    (N, 1, 1))                                       # (N, n_m, 2)
    z_pos = np.tile(np.asarray(state.zealot_positions, dtype=np.float64)[None],
                    (N, 1, 1))                                       # (N, n_z, 2)
    m_hp = np.tile(np.asarray(state.marine_hps, dtype=np.float64)[None],
                   (N, 1))                                           # (N, n_m)
    z_hp = np.tile(np.asarray(state.zealot_hps, dtype=np.float64)[None],
                   (N, 1))                                           # (N, n_z)
    z_alive = np.tile(np.asarray(state.zealot_alive, dtype=bool)[None],
                      (N, 1))                                        # (N, n_z)

    out_m_pos = np.zeros((N, H, n_m, 2))
    out_z_pos = np.zeros((N, H, n_z, 2))
    out_m_hp = np.zeros((N, H, n_m))
    out_z_hp = np.zeros((N, H, n_z))
    out_z_alive = np.zeros((N, H, n_z), dtype=bool)
    out_m_ready = np.zeros((N, H, n_m), dtype=bool)

    arange_N = np.arange(N)
    arange_N_col = arange_N[:, None]

    for t in range(H):
        u = actions[:, :, t, :]                                      # (N, n_m, 2)
        moving = np.linalg.norm(u, axis=-1) > 0.1                    # (N, n_m)

        if stochastic:
            m_pos = m_pos + u * MARINE_SPEED * dt \
                    + np.random.normal(0, MARINE_EXEC_NOISE, (N, n_m, 2))
        else:
            m_pos = m_pos + u * MARINE_SPEED * dt

        m_alive = m_hp > 0                                            # (N, n_m)

        # Zealot target picking — softmax over (-k * d_to_marine)
        z2m_diffs = m_pos[:, None, :, :] - z_pos[:, :, None, :]       # (N, n_z, n_m, 2)
        z2m_d = np.linalg.norm(z2m_diffs, axis=-1)                    # (N, n_z, n_m)
        z2m_d = np.where(m_alive[:, None, :], z2m_d, np.inf)

        if stochastic:
            logits = -ZEALOT_SWITCH_SHARPNESS * z2m_d
            # Gumbel-max sampling for stable categorical sampling
            gumbels = -np.log(-np.log(np.random.uniform(1e-12, 1.0, (N, n_z, n_m))))
            target = np.argmax(logits + gumbels, axis=-1)             # (N, n_z)
        else:
            target = np.argmin(z2m_d, axis=-1)

        # Vector from each zealot to its target
        target_pos = m_pos[arange_N_col, target]                      # (N, n_z, 2)
        zr = target_pos - z_pos                                       # (N, n_z, 2)
        zd = np.linalg.norm(zr, axis=-1, keepdims=True)               # (N, n_z, 1)
        zdir = np.where(zd > 0.1, zr / np.maximum(zd, 1e-6), 0.0)     # (N, n_z, 2)

        if stochastic:
            ang = np.random.normal(0, ZEALOT_PURSUIT_NOISE, (N, n_z))
            ca = np.cos(ang)
            sa = np.sin(ang)
            zdir = np.stack([
                ca * zdir[..., 0] - sa * zdir[..., 1],
                sa * zdir[..., 0] + ca * zdir[..., 1],
            ], axis=-1)

        move_d = np.minimum(ZEALOT_SPEED * dt, zd[..., 0])            # (N, n_z)
        z_step = zdir * move_d[..., None]
        z_pos = z_pos + z_step * z_alive[..., None]

        # Marine shooting — closest live zealot in range, stationary marine
        m2z_diffs = z_pos[:, None, :, :] - m_pos[:, :, None, :]       # (N, n_m, n_z, 2)
        m2z_d = np.linalg.norm(m2z_diffs, axis=-1)
        m2z_d = np.where(z_alive[:, None, :], m2z_d, np.inf)
        m_target = np.argmin(m2z_d, axis=-1)                          # (N, n_m)
        m_target_d = np.min(m2z_d, axis=-1)
        shoots = m_alive & (~moving) & (m_target_d <= MARINE_RANGE)   # (N, n_m)

        damage = shoots * MARINE_DPS * dt
        m_b = np.broadcast_to(arange_N_col, (N, n_m))
        np.add.at(z_hp, (m_b, m_target), -damage)

        # Zealot melee damage — its current target if within 1.0
        new_zr = m_pos[arange_N_col, target] - z_pos
        new_zd = np.linalg.norm(new_zr, axis=-1)
        in_melee = (new_zd < 1.0) & z_alive
        z_dmg = in_melee * ZEALOT_DPS * dt
        z_b = np.broadcast_to(arange_N_col, (N, n_z))
        np.add.at(m_hp, (z_b, target), -z_dmg)

        z_alive = z_alive & (z_hp > 0)
        z_hp = np.maximum(z_hp, 0.0)
        m_hp = np.maximum(m_hp, 0.0)

        out_m_pos[:, t] = m_pos
        out_z_pos[:, t] = z_pos
        out_m_hp[:, t] = m_hp
        out_z_hp[:, t] = z_hp
        out_z_alive[:, t] = z_alive
        out_m_ready[:, t] = ~moving

    return {
        'm_pos': out_m_pos, 'z_pos': out_z_pos,
        'm_hp': out_m_hp, 'z_hp': out_z_hp,
        'z_alive': out_z_alive, 'm_ready': out_m_ready,
    }


def mpc_select_action_vectorized(state, n_candidates=128, n_scenarios=6,
                                  horizon=8, dt=0.4, cvar_alpha=0.3):
    """Fully vectorized SMPC for the circle scenario.

    Returns:
      best_actions: list of n_marines np.ndarray(2) — first action per marine
      best_components: dict[str, float] — per-component cost from median
                        scenario of best candidate
    """
    n_m = state.n_marines
    n_z = state.n_zealots
    C = n_candidates
    S = n_scenarios

    # Sample C candidate action sequences (deterministic per candidate)
    actions_c = sample_actions_batch(state, C, horizon)               # (C, n_m, H, 2)

    # Replicate each candidate S times for stochastic scenarios
    actions = np.repeat(actions_c, S, axis=0)                         # (C*S, n_m, H, 2)

    # Simulate
    traj = simulate_batch(state, actions, dt=dt, stochastic=True)

    # Cost
    total, comps = compute_cost_batch(traj, n_m, n_z)                 # total (C*S,)

    total_2d = total.reshape(C, S)
    sorted_costs = np.sort(total_2d, axis=1)[:, ::-1]                 # worst → best
    n_tail = max(1, int(np.ceil(S * cvar_alpha)))
    cvar = np.mean(sorted_costs[:, :n_tail], axis=1)                  # (C,)

    best_idx = int(np.argmin(cvar))
    best_actions = [actions_c[best_idx, m, 0, :].copy() for m in range(n_m)]

    # Median-scenario components for the chosen candidate (for logging)
    comps_2d = {k: v.reshape(C, S) for k, v in comps.items()}
    sorted_idx = np.argsort(total_2d[best_idx])
    median_idx = int(sorted_idx[S // 2])
    best_components = {k: float(comps_2d[k][best_idx, median_idx]) for k in comps_2d}

    return best_actions, best_components
