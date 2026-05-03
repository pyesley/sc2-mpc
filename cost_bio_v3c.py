"""
Bio cost — variant v3c: HOLD-LITE.

Hypothesis: v2c's 32% HOLD bias produced 1 timeout (bio held forever).
Back off slightly (HOLD 24%) and shift the freed mass to SHIELD.
Tests whether the v2c gain comes from "more hold" or just from
"less independent / more coordinated".
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch

mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.30, 0.24])
