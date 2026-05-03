"""
Bio cost — variant v1a: SURVIVAL-FIRST.

Hypothesis: 0% baseline win rate is because matchup costs (45k/game)
dominate survival (97k/game shows units constantly losing HP). Marines
walk into stalker fire because the cost of being out-of-position is
higher than the cost of dying.

Edits (vs cost_bio defaults):
  - Per-unit-death penalties roughly 3× stronger
  - Per-HP-loss survival weights 2× stronger
  - Matchup multipliers halved (less push to engage at unsafe ranges)
  - medivac_safety halved (medivac MUST be near bio = inside stalker
    range to heal; over-penalizing safety stops it from healing)
"""

from cost_bio import compute_cost, compute_cost_batch
import cost_bio

cost_bio.W_SURV_M = 12.0
cost_bio.W_SURV_MM = 16.0
cost_bio.W_SURV_MV = 20.0
cost_bio.W_DEAD_M = 24.0
cost_bio.W_DEAD_MM = 42.0
cost_bio.W_DEAD_MV = 120.0

cost_bio.W_MATCHUP_M = 0.5
cost_bio.W_MATCHUP_MM = 0.5
cost_bio.W_MEDIVAC_SAFETY = 0.5
