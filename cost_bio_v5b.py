"""v5b: v2c + h=10. Slightly longer than 8 — see if there's a sweet
spot between v2c (h=8, 75%) and v4a (h=12, 8%)."""
import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio, mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])
mpc_vectorized_bio.MPC_HORIZON = 10
