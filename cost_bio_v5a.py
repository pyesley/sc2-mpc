"""v5a: v2c + SHORTER horizon (h=6). Test whether sim accuracy improves
when MPC plans only 2.4 s ahead instead of 3.2 s."""
import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio, mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])
mpc_vectorized_bio.MPC_HORIZON = 6
