"""
Type-pair matchup primitives — the building blocks of compositional cost.

Each primitive takes positions / HPs / distance arrays and returns a
cost contribution. Higher-level cost functions (cost_bio, future
cost_zerg, cost_protoss, ...) compose these primitives across whatever
army composition is on the field.

All primitives operate on numpy arrays so they vectorize naturally:
  - distance arg `d` is shape () or (N,) or (N, n_a) etc.
  - return value has the same shape as `d`

Matchup library (positional cost between one of mine and one of theirs):

  kite_marine_vs_zealot     marine 5 / zealot 1 (melee) → kite at 3-5
  engage_marine_vs_stalker  marine 5 / stalker 6 → MUST close, can't kite
  kite_marauder_vs_zealot   marauder 6 (slow) / zealot 1 → kite at 4-6
  engage_marauder_vs_stalker marauder 6 / stalker 6 → equal range, dance
  medivac_safety            air, no weapon / range-6 stalker threat
  medivac_heal_proximity    medivac heal range 4 → stay near injured bio

Globals (universal across compositions):

  focus_fire_bonus          reward concentrating fire on weakest target
  army_cohesion             reward bio-army cluster small enough for medivac
"""

import numpy as np


# ─── Game constants (canonical SC2 values) ────────────────────
MARINE_RANGE = 5.0
MARAUDER_RANGE = 6.0
STALKER_RANGE = 6.0
MEDIVAC_HEAL_RANGE = 4.0

MARINE_HP_MAX = 45.0
MARAUDER_HP_MAX = 125.0
MEDIVAC_HP_MAX = 150.0
ZEALOT_HP_MAX = 100.0 + 50.0       # hp + shield
STALKER_HP_MAX = 80.0 + 80.0


# ─── Distance helper (batch-friendly) ─────────────────────────
def pairwise_dist(pos_a, pos_b):
    """pos_a (..., n_a, 2), pos_b (..., n_b, 2) → (..., n_a, n_b)."""
    diffs = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    return np.linalg.norm(diffs, axis=-1)


# ─── Per-pair matchup primitives ──────────────────────────────
def kite_marine_vs_zealot(d):
    """Marine should kite zealot at ~3.5-5.0 distance.
    melee (d<1.5) = death; in MARINE_RANGE = good."""
    return np.where(d < 1.5, 20.0 * (1.5 - d),
           np.where(d < 3.0,  4.0 * (3.0 - d),
           np.where(d > 7.0,  1.5 * (d - 7.0), 0.0)))


def engage_marine_vs_stalker(d):
    """Marine vs stalker: stalker outranges (6 vs 5), so marine MUST
    close to MARINE_RANGE. No kiting."""
    return np.where(d > MARINE_RANGE, 4.0 * (d - MARINE_RANGE),
           np.where(d < 3.5, 0.0, -1.0))   # in range = mild reward


def kite_marauder_vs_zealot(d):
    """Marauder is slower than marine but has range 6. Kite zealot at
    ~4-6 distance."""
    return np.where(d < 1.5, 25.0 * (1.5 - d),
           np.where(d < 3.5, 4.0 * (3.5 - d),
           np.where(d > 8.0, 1.5 * (d - 8.0), 0.0)))


def engage_marauder_vs_stalker(d):
    """Marauder vs stalker: equal range, marauder has +armored bonus
    damage. Hold at MARAUDER_RANGE edge."""
    return np.where(d > MARAUDER_RANGE, 4.0 * (d - MARAUDER_RANGE),
           np.where(d < 4.5, 0.0, -2.0))   # in range = stronger reward (priority target)


# ─── Medivac primitives ───────────────────────────────────────
def medivac_safety(d_to_nearest_enemy):
    """Medivac is air, no weapon, but stalkers shoot it (range 6).
    Penalize being inside any threat range."""
    return np.where(d_to_nearest_enemy < STALKER_RANGE,
                    8.0 * (STALKER_RANGE - d_to_nearest_enemy), 0.0)


def medivac_heal_proximity(d_to_most_injured_ally):
    """Reward being inside heal range of the most-injured bio ally."""
    return np.where(d_to_most_injured_ally < MEDIVAC_HEAL_RANGE, -4.0,
           np.where(d_to_most_injured_ally < 8.0,
                    1.0 * (d_to_most_injured_ally - MEDIVAC_HEAL_RANGE),
                    4.0 + 2.0 * (d_to_most_injured_ally - 8.0)))


# ─── Universal globals ────────────────────────────────────────
def focus_fire_bonus(n_attackers_in_range_of_weakest):
    """Reward concentrating fire on the lowest-HP enemy. Each attacker
    in range gives a small reward."""
    return -2.0 * n_attackers_in_range_of_weakest


def survival_cost(total_my_hp, total_my_hp_max, n_dead, dead_penalty=15.0,
                  total_dead_penalty=80.0, n_units=1):
    """Generic survival: linear in HP fraction, with extra per-death
    penalty and a large all-dead penalty."""
    if n_dead == n_units:
        return total_dead_penalty
    return 8.0 * (1.0 - total_my_hp / max(total_my_hp_max, 1e-6)) + dead_penalty * n_dead


def enemy_hp_drive(total_enemy_hp, total_enemy_hp_max, win_bonus=-15.0):
    """Drive damage to enemies."""
    if total_enemy_hp <= 0:
        return win_bonus
    return 12.0 * (total_enemy_hp / max(total_enemy_hp_max, 1e-6))
