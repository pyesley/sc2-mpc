"""
Bio cost — variant v1c: FOCUS-FIRE DOMINATES.

Hypothesis: bio loses because damage spreads across both stalkers
instead of bursting one down fast. If we crank focus_fire (concentrate
all in-range bio onto the lowest-HP enemy) and stalker_priority
(strongly bias on getting bio inside marauder range of every alive
stalker), we kill stalkers first, then have a free kite-fight on the
zealots.

Edits (vs cost_bio defaults):
  - W_FOCUS_FIRE 2.5 → 10.0  (4× per-attacker-on-weakest reward)
  - W_STALKER_PRIORITY 3.0 → 8.0
  - W_DPS_M / W_DPS_MM also 1.5× to bias bio toward shooting over moving
"""

from cost_bio import compute_cost, compute_cost_batch
import cost_bio

cost_bio.W_FOCUS_FIRE = 10.0
cost_bio.W_STALKER_PRIORITY = 8.0
cost_bio.W_DPS_M = 3.0
cost_bio.W_DPS_MM = 4.0
