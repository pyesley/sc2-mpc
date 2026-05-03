"""
Bio cost — variant v3a: HOLD + FORMATION-LOCK.

Combines v2c's hold-bias mode mix with v2a's stronger marauder_front
penalty. Hypothesis: holding lets formation set up; the heavy
marauder_front weight then locks it in.
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch
import cost_bio_v1b

cost_bio_v1b.W_MARAUDER_FRONT = 14.0
cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch

# v2c's mode mix
mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])
