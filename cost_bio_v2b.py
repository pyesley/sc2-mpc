"""
Bio cost — variant v2b: PROTECT-MARINES.

Builds on v1b. Hypothesis: marines are 6/8 of our DPS and 45 HP each
(cheap individually but the swarm is the army's damage). Current
weights value enemy_hp damage more than marine-death prevention. Make
losing a marine HURT.

Edits vs v1b:
  - W_DEAD_M     8.0 → 24.0   (per-marine death: 3×)
  - W_SURV_M     6.0 → 14.0   (HP loss across remaining marines)
  - W_MATCHUP_M  1.0 → 0.6    (matchup pulls marines into stalker fire,
                                tone it down so retreat is preferred)
  - keeps v1b's marauder_front hook at W=4
"""

from cost_bio import compute_cost, compute_cost_batch
import cost_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.W_DEAD_M = 24.0
cost_bio.W_SURV_M = 14.0
cost_bio.W_MATCHUP_M = 0.6

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
