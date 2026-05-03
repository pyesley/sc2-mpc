"""
Round 29 cost — variant v1a: KILL-PUSH (state-conditional).

Hypothesis: 14% of baseline runs are TIMEOUTs where one zealot is left
at very low HP but marines stay too far to close. If we detect "killing
mode" (total alive zealot HP < 100) and *invert* the kite distance — pull
the marines toward the surviving zealot to finish it — we should kill
those timeouts.

Diff vs cost_circle.py:
  - new flag        killing_mode = (total_alive_zealot_hp < 100)
  - kiter_distance  in killing mode rewards d <= MARINE_RANGE, penalizes
                    d > MARINE_RANGE; baseline behavior outside.
  - new component   kill_push : -5 per alive marine within MARINE_RANGE
                    of any alive zealot, only in killing mode.
"""

import numpy as np
from typing import Tuple, Dict

MARINE_RANGE = 5.0
ZEALOT_HP_MAX = 150.0
MARINE_HP_MAX = 45.0
KILL_THRESHOLD = 100.0   # total alive zealot HP below which "kill push" activates


def compute_cost(state) -> Tuple[float, Dict[str, float]]:
    components: Dict[str, float] = {}

    n = state.n_marines
    live = [i for i in range(n) if state.marine_hps[i] > 0]
    n_alive = len(live)

    alive_z = [j for j in range(state.n_zealots) if state.zealot_alive[j]]
    n_z = len(alive_z)

    if n_z > 0:
        total_z_hp = sum(state.zealot_hps[j] for j in alive_z)
        components['zealot_hp'] = 12.0 * (total_z_hp / (state.n_zealots * ZEALOT_HP_MAX))
    else:
        components['zealot_hp'] = -15.0
        total_z_hp = 0.0

    if n_alive == 0:
        components['marine_survival'] = 80.0
    else:
        total_hp = sum(state.marine_hps[i] for i in live)
        components['marine_survival'] = 8.0 * (1.0 - total_hp / (n * MARINE_HP_MAX))
        components['marine_survival'] += 15.0 * (n - n_alive)

    if n_alive == 0 or n_z == 0:
        for k in ('shooter_in_range', 'kiter_distance', 'aggro_split',
                  'kiter_opposite', 'marine_spread', 'dps_uptime', 'kill_push'):
            components[k] = 0.0
        return sum(components.values()), components

    # ── State-conditional flag ──
    killing_mode = total_z_hp < KILL_THRESHOLD

    closest_z_dist = []
    for i in live:
        d = min(np.linalg.norm(state.marine_positions[i] - state.zealot_positions[j])
                for j in alive_z)
        closest_z_dist.append(d)

    shooter_local = int(np.argmax(closest_z_dist))
    shooter_idx = live[shooter_local]
    shooter_dist = closest_z_dist[shooter_local]
    kiter_idxs = [live[k] for k in range(n_alive) if k != shooter_local]
    kiter_dists = [closest_z_dist[k] for k in range(n_alive) if k != shooter_local]

    if shooter_dist > MARINE_RANGE:
        components['shooter_in_range'] = 4.0 * (shooter_dist - MARINE_RANGE)
    elif shooter_dist < 2.0:
        components['shooter_in_range'] = 10.0 * (2.0 - shooter_dist)
    else:
        components['shooter_in_range'] = -3.0

    # ── EDIT: kiter_distance switches to "pull-in" in killing mode ──
    kite_cost = 0.0
    if killing_mode:
        # Reward kiters being IN firing range; penalize being out.
        for d in kiter_dists:
            if d > MARINE_RANGE:
                kite_cost += 3.0 * (d - MARINE_RANGE)
            elif d > 3.0:
                kite_cost += -1.0          # mild reward for being in range
            # d <= 3.0: no cost (close enough to shoot, melee less of a threat
            #                    when only one zealot is left)
    else:
        # Baseline kite behavior
        for d in kiter_dists:
            if d < 1.5:
                kite_cost += 20.0 * (1.5 - d)
            elif d < 3.0:
                kite_cost += 4.0 * (3.0 - d)
            elif d > 7.0:
                kite_cost += 1.5 * (d - 7.0)
    components['kiter_distance'] = kite_cost

    # ── NEW: kill_push reward (in range marines when killing mode) ──
    if killing_mode:
        in_range = sum(1 for d in closest_z_dist if d <= MARINE_RANGE)
        components['kill_push'] = -5.0 * in_range
    else:
        components['kill_push'] = 0.0

    if n_z >= 2 and n_alive >= 2:
        closest_marine = []
        for j in alive_z:
            m = min(live, key=lambda i: np.linalg.norm(
                state.zealot_positions[j] - state.marine_positions[i]))
            closest_marine.append(m)
        distinct = len(set(closest_marine))
        if distinct == 1:
            components['aggro_split'] = 8.0
        else:
            components['aggro_split'] = -3.0
    else:
        components['aggro_split'] = 0.0

    if len(kiter_idxs) == 2:
        sh_pos = state.marine_positions[shooter_idx]
        v1 = state.marine_positions[kiter_idxs[0]] - sh_pos
        v2 = state.marine_positions[kiter_idxs[1]] - sh_pos
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 0.5 and n2 > 0.5:
            cos_angle = float(np.dot(v1, v2) / (n1 * n2))
            components['kiter_opposite'] = 4.0 * (cos_angle + 1.0)
        else:
            components['kiter_opposite'] = 0.0
    else:
        components['kiter_opposite'] = 0.0

    if n_alive >= 2:
        min_sep = float('inf')
        for a_idx in range(n_alive):
            for b_idx in range(a_idx + 1, n_alive):
                sep = np.linalg.norm(
                    state.marine_positions[live[a_idx]] -
                    state.marine_positions[live[b_idx]])
                min_sep = min(min_sep, sep)
        if min_sep < 2.0:
            components['marine_spread'] = 4.0 * (2.0 - min_sep)
        else:
            components['marine_spread'] = 0.0
    else:
        components['marine_spread'] = 0.0

    dps_reward = 0.0
    for k_idx, i in enumerate(live):
        if state.marine_weapon_ready[i] and closest_z_dist[k_idx] <= MARINE_RANGE:
            dps_reward -= 3.0
    components['dps_uptime'] = dps_reward

    return sum(components.values()), components


def compute_cost_batch(traj, n_marines: int, n_zealots: int,
                       marine_hp_max: float = MARINE_HP_MAX,
                       zealot_hp_max: float = ZEALOT_HP_MAX):
    m_pos = traj['m_pos']
    z_pos = traj['z_pos']
    m_hp = traj['m_hp']
    z_hp = traj['z_hp']
    z_alive = traj['z_alive']
    m_ready = traj['m_ready']

    N, H, n_m, _ = m_pos.shape
    n_z = z_pos.shape[2]
    assert n_m == n_marines and n_z == n_zealots

    keys = ('zealot_hp', 'marine_survival', 'shooter_in_range',
            'kiter_distance', 'aggro_split', 'kiter_opposite',
            'marine_spread', 'dps_uptime', 'kill_push')   # added kill_push
    comps = {k: np.zeros(N) for k in keys}
    arange_N = np.arange(N)

    err_ctx = np.errstate(invalid='ignore', over='ignore')
    err_ctx.__enter__()
    for t in range(H):
        mp = m_pos[:, t]
        zp = z_pos[:, t]
        mh = m_hp[:, t]
        zh = z_hp[:, t]
        za = z_alive[:, t]
        mr = m_ready[:, t]

        m_alive = mh > 0
        n_m_alive = m_alive.sum(axis=-1)
        n_z_alive = za.sum(axis=-1)
        all_m_dead = n_m_alive == 0
        all_z_dead = n_z_alive == 0
        active = ~(all_m_dead | all_z_dead)

        z_hp_eff = (zh * za).sum(axis=-1)                  # (N,) total alive zealot HP
        zealot_hp_cost = 12.0 * (z_hp_eff / (n_zealots * zealot_hp_max))
        zealot_hp_cost = np.where(all_z_dead, -15.0, zealot_hp_cost)
        comps['zealot_hp'] += zealot_hp_cost

        m_hp_sum = (mh * m_alive).sum(axis=-1)
        n_dead = n_marines - n_m_alive
        ms_cost = np.where(
            all_m_dead, 80.0,
            8.0 * (1.0 - m_hp_sum / (n_marines * marine_hp_max)) + 15.0 * n_dead
        )
        comps['marine_survival'] += ms_cost

        # State-conditional killing-mode flag
        killing = active & (z_hp_eff < KILL_THRESHOLD) & (n_z_alive > 0)

        diffs = mp[:, :, None, :] - zp[:, None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        dists_mask = np.where(za[:, None, :], dists, np.inf)
        closest_z_dist = np.min(dists_mask, axis=-1)
        czd_alive = np.where(m_alive, closest_z_dist, -np.inf)
        shooter_idx = np.argmax(czd_alive, axis=-1)
        shooter_dist = czd_alive[arange_N, shooter_idx]

        sir = np.where(
            shooter_dist > MARINE_RANGE,
            4.0 * (shooter_dist - MARINE_RANGE),
            np.where(shooter_dist < 2.0,
                     10.0 * (2.0 - shooter_dist),
                     -3.0)
        )
        sir = np.where(active & (n_m_alive > 0), sir, 0.0)
        comps['shooter_in_range'] += sir

        is_shooter = (np.arange(n_m)[None, :] == shooter_idx[:, None])
        is_kiter = m_alive & ~is_shooter
        d = closest_z_dist
        # ── EDIT: kiter_distance has two formulas, selected by killing flag ──
        kite_normal = np.where(
            d < 1.5, 20.0 * (1.5 - d),
            np.where(d < 3.0, 4.0 * (3.0 - d),
                     np.where(d > 7.0, 1.5 * (d - 7.0), 0.0))
        )
        kite_kill = np.where(
            d > MARINE_RANGE, 3.0 * (d - MARINE_RANGE),
            np.where(d > 3.0, -1.0, 0.0)
        )
        kite_per_m = np.where(killing[:, None], kite_kill, kite_normal)
        kd = (kite_per_m * is_kiter).sum(axis=-1)
        kd = np.where(active, kd, 0.0)
        comps['kiter_distance'] += kd

        # ── NEW: kill_push — reward marines in range when killing ──
        in_range_alive = (closest_z_dist <= MARINE_RANGE) & m_alive
        in_range_count = in_range_alive.sum(axis=-1)
        kp = np.where(killing, -5.0 * in_range_count, 0.0)
        comps['kill_push'] += kp

        z2m_diffs = mp[:, None, :, :] - zp[:, :, None, :]
        z2m_dists = np.linalg.norm(z2m_diffs, axis=-1)
        z2m_mask = np.where(m_alive[:, None, :], z2m_dists, np.inf)
        closest_m_per_z = np.argmin(z2m_mask, axis=-1)
        if n_zealots == 2:
            both_alive = za[:, 0] & za[:, 1]
            same_target = closest_m_per_z[:, 0] == closest_m_per_z[:, 1]
            same_aggro = both_alive & same_target
            aggro = np.where(same_aggro, 8.0,
                             np.where(both_alive, -3.0, 0.0))
        else:
            aggro = np.zeros(N)
        aggro = np.where(active, aggro, 0.0)
        comps['aggro_split'] += aggro

        if n_marines == 3:
            kiter_pair = np.array([[1, 2], [0, 2], [0, 1]])
            kiter_idxs = kiter_pair[shooter_idx]
            sh_pos = mp[arange_N, shooter_idx]
            ka_pos = mp[arange_N, kiter_idxs[:, 0]]
            kb_pos = mp[arange_N, kiter_idxs[:, 1]]
            v1 = ka_pos - sh_pos
            v2 = kb_pos - sh_pos
            n1 = np.linalg.norm(v1, axis=-1)
            n2 = np.linalg.norm(v2, axis=-1)
            valid = (n1 > 0.5) & (n2 > 0.5)
            cos_angle = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-8)
            ko = 4.0 * (cos_angle + 1.0)
            both_kiters_alive = m_alive[arange_N, kiter_idxs[:, 0]] & \
                                m_alive[arange_N, kiter_idxs[:, 1]]
            ko = np.where(valid & both_kiters_alive, ko, 0.0)
        else:
            ko = np.zeros(N)
        ko = np.where(active, ko, 0.0)
        comps['kiter_opposite'] += ko

        if n_marines == 3:
            d01 = np.linalg.norm(mp[:, 0] - mp[:, 1], axis=-1)
            d02 = np.linalg.norm(mp[:, 0] - mp[:, 2], axis=-1)
            d12 = np.linalg.norm(mp[:, 1] - mp[:, 2], axis=-1)
            d01 = np.where(m_alive[:, 0] & m_alive[:, 1], d01, np.inf)
            d02 = np.where(m_alive[:, 0] & m_alive[:, 2], d02, np.inf)
            d12 = np.where(m_alive[:, 1] & m_alive[:, 2], d12, np.inf)
            min_sep = np.minimum(np.minimum(d01, d02), d12)
            spread = np.where(min_sep < 2.0, 4.0 * (2.0 - min_sep), 0.0)
            spread = np.where(np.isinf(min_sep), 0.0, spread)
        else:
            spread = np.zeros(N)
        spread = np.where(active & (n_m_alive >= 2), spread, 0.0)
        comps['marine_spread'] += spread

        in_range = closest_z_dist <= MARINE_RANGE
        dps_per_m = (m_alive & mr & in_range) * -3.0
        dps = dps_per_m.sum(axis=-1)
        dps = np.where(active, dps, 0.0)
        comps['dps_uptime'] += dps

    err_ctx.__exit__(None, None, None)
    total = sum(comps.values())
    return total, comps
