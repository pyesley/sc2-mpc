"""
Bio cost — variant v6a: v2c + COHESION (anti-split).

Combines v2c (HOLD-heavy mode mix) with the new cohesion cost.
Hypothesis (from observed games): the enemy splits the bio army by
chasing different marines in different directions; once split, each
unit gets isolated and killed. A cost that explicitly penalizes the
army being stretched out should counter this.

Threshold 6.0 — chosen so a valid SHIELD formation (marauders ~5
units forward of marines) does NOT trigger the penalty, but a
straggler 8+ units away does.
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])

# Turn cohesion on (default in cost_bio is 0.0)
cost_bio.W_COHESION = 5.0
