"""v5c: v2c + boost focus_fire (W_FOCUS_FIRE 2.5 → 8.0).
Concentrates damage on lowest-HP enemy. Maybe the issue isn't formation
but that bio splits damage and never finishes a kill. Combine with v2c's
hold mode."""
import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio, mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch
mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])

cost_bio.W_FOCUS_FIRE = 8.0
